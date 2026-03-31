"""
Retry logic with exponential backoff for web crawling.

Provides resilient crawling with automatic retries, exponential backoff,
and circuit breaker pattern for failed sources.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

import httpx
from crawl4ai import AsyncWebCrawler


T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_status_codes: list[int] = None
    retry_on_exceptions: list[type] = None
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300  # 5 minutes
    
    def __post_init__(self):
        if self.retry_on_status_codes is None:
            self.retry_on_status_codes = [408, 429, 500, 502, 503, 504]
        if self.retry_on_exceptions is None:
            self.retry_on_exceptions = [
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
                asyncio.TimeoutError,
            ]


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing fast, no requests allowed
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, threshold: int = 5, timeout: int = 300):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        async with self._lock:
            self.failures = 0
            self.state = "CLOSED"
    
    async def _record_failure(self):
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryableCrawler:
    """
    Web crawler with automatic retries and exponential backoff.
    
    Features:
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Status code-based retries
    - Exception-based retries
    - Per-URL retry tracking
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_counts: dict[str, int] = {}
        
    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """Get or create circuit breaker for URL."""
        domain = self._extract_domain(url)
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = CircuitBreaker(
                threshold=self.config.circuit_breaker_threshold,
                timeout=self.config.circuit_breaker_timeout
            )
        return self.circuit_breakers[domain]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        else:  # EXPONENTIAL
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Apply max delay cap
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def _should_retry_status_code(self, status_code: int) -> bool:
        """Check if status code should trigger retry."""
        return status_code in self.config.retry_on_status_codes
    
    def _should_retry_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger retry."""
        return any(isinstance(exception, exc_type) 
                  for exc_type in self.config.retry_on_exceptions)
    
    async def crawl_with_retry(
        self, 
        url: str, 
        crawler: AsyncWebCrawler = None,
        extractor: Callable = None
    ) -> Any:
        """
        Crawl URL with automatic retries.
        
        Args:
            url: URL to crawl
            crawler: AsyncWebCrawler instance (created if None)
            extractor: Optional function to extract data from result
        
        Returns:
            Crawl result or extracted data
        """
        circuit_breaker = self._get_circuit_breaker(url)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Check circuit breaker
                result = await circuit_breaker.call(
                    self._do_crawl, url, crawler, extractor
                )
                
                # Track success
                self.retry_counts[url] = attempt
                return result
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    print(f"[RetryCrawler] Failed after {attempt + 1} attempts: {url}")
                    raise
                
                # Check if we should retry
                if isinstance(e, CircuitBreakerOpenError):
                    print(f"[RetryCrawler] Circuit breaker open for {self._extract_domain(url)}, skipping...")
                    raise
                
                if not self._should_retry_exception(e):
                    raise
                
                # Calculate and apply delay
                delay = self._calculate_delay(attempt)
                print(f"[RetryCrawler] Attempt {attempt + 1} failed for {url}: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        return None
    
    async def _do_crawl(
        self, 
        url: str, 
        crawler: AsyncWebCrawler,
        extractor: Callable = None
    ) -> Any:
        """Perform actual crawl."""
        if crawler is None:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
        else:
            result = await crawler.arun(url=url)
        
        if extractor:
            return extractor(result)
        return result
    
    async def fetch_with_retry(
        self, 
        url: str,
        client: httpx.AsyncClient = None,
        **kwargs
    ) -> httpx.Response:
        """
        HTTP fetch with automatic retries.
        
        Args:
            url: URL to fetch
            client: httpx client (created if None)
            **kwargs: Additional arguments for httpx request
        """
        circuit_breaker = self._get_circuit_breaker(url)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if client is None:
                    async with httpx.AsyncClient() as client:
                        response = await circuit_breaker.call(
                            self._do_fetch, url, client, **kwargs
                        )
                else:
                    response = await circuit_breaker.call(
                        self._do_fetch, url, client, **kwargs
                    )
                
                # Check status code
                if self._should_retry_status_code(response.status_code):
                    raise httpx.HTTPStatusError(
                        f"Status {response.status_code}",
                        request=response.request,
                        response=response
                    )
                
                self.retry_counts[url] = attempt
                return response
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    print(f"[RetryCrawler] HTTP failed after {attempt + 1} attempts: {url}")
                    raise
                
                if isinstance(e, CircuitBreakerOpenError):
                    raise
                
                if isinstance(e, httpx.HTTPStatusError):
                    if not self._should_retry_status_code(e.response.status_code):
                        raise
                elif not self._should_retry_exception(e):
                    raise
                
                delay = self._calculate_delay(attempt)
                print(f"[RetryCrawler] HTTP attempt {attempt + 1} failed for {url}: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        return None
    
    async def _do_fetch(self, url: str, client: httpx.AsyncClient, **kwargs) -> httpx.Response:
        """Perform actual HTTP fetch."""
        return await client.get(url, **kwargs)
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        total_requests = len(self.retry_counts)
        total_retries = sum(self.retry_counts.values())
        avg_retries = total_retries / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "total_retries": total_retries,
            "average_retries_per_request": avg_retries,
            "circuit_breaker_states": {
                domain: cb.state 
                for domain, cb in self.circuit_breakers.items()
            },
        }


# Convenience decorator
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
):
    """Decorator to add retry logic to any async function."""
    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy
        )
        crawler = RetryableCrawler(config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    
                    delay = crawler._calculate_delay(attempt)
                    print(f"[Retry] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            
            return None
        
        return wrapper
    return decorator


# Usage examples
async def test_retry_crawler():
    """Test the retry crawler."""
    config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        exponential_base=2.0,
        strategy=RetryStrategy.EXPONENTIAL
    )
    
    crawler = RetryableCrawler(config)
    
    # Test with a real URL
    test_urls = [
        "https://www.iitb.ac.in/sine",  # Should succeed
        "https://this-domain-does-not-exist-12345.com",  # Should fail
    ]
    
    for url in test_urls:
        try:
            print(f"\nTesting: {url}")
            result = await crawler.crawl_with_retry(url)
            print(f"Success! Content length: {len(str(result)) if result else 0}")
        except Exception as e:
            print(f"Failed: {e}")
    
    # Print stats
    stats = crawler.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_retry_crawler())
