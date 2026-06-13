import httpx


class AsyncHTTPClientPool:
    """Пул переиспользуемых 
    асинхронных HTTPX-клиентов.
    Обеспечивает удержание keep-alive 
    соединений для  снижения сетевого 
    оверхеда при частых вызовах LLM.
    """

    __slots__ = (
        "_closed", 
        "_direct", 
        "_proxy"
    )
    

    def __init__(
        self,
        proxy_url: str | None = None
    ) -> None:
        
        # Хардкодим 
        # оптимальные таймауты 
        timeout = httpx.Timeout(
            timeout=120.0,
            connect=15.0,
            write=120.0,
            read=120.0,
            pool=15.0,
        )
        
        # Лимиты пула для
        # защиты от исчерпания 
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        )
        
        # Инициализируем 
        # основной прямой клиент
        self._direct = httpx.AsyncClient(
            timeout=timeout, 
            limits=limits, 
            http2=True
        )
        
        # Инициализируем 
        # клиент для прокси
        self._proxy = (
            httpx.AsyncClient(
                timeout=timeout, 
                limits=limits, 
                proxy=proxy_url, 
                http2=True
            ) if proxy_url else self._direct
        )
        
        self._closed = False




    def client(
        self, 
        use_proxy: bool
    ) -> httpx.AsyncClient:
        """Мгновенно отдает 
        нужный инстанс клиента."""
        
        return (
            self._proxy 
            if use_proxy 
            else self._direct
        )
    
    


    async def aclose(self) -> None:
        """Безопасно и идемпотентно 
        закрывает соединения."""
        
        if self._closed:
            return
            
        self._closed = True
        await self._direct.aclose()
        
        # Освобождаем прокси-клиент, 
        # если это отдельный инстанс
        if self._proxy is not self._direct:
            await self._proxy.aclose()