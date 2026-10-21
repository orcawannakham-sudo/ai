"""On-demand web search with HTML and RSS fallbacks using the standard library."""

from __future__ import annotations

import ipaddress
import re
from html import unescape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from xml.etree import ElementTree


SEARCH_ENDPOINTS = (
    ("DuckDuckGo HTML", "https://html.duckduckgo.com/html/", "html"),
    ("DuckDuckGo Lite", "https://lite.duckduckgo.com/lite/", "html"),
    ("Bing RSS", "https://www.bing.com/search", "rss"),
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
MAX_QUERY_LENGTH = 300
MAX_SOURCE_BYTES = 2_000_000
CODE_REQUEST_RE = re.compile(
    r"\b(code|example|program|script|snippet|calculator)\b", re.IGNORECASE
)
SNIPPET_CODE_PATTERNS = (
    re.compile(r"`([^`\n]{3,300})`"),
    re.compile(r"\bprint\s*\([^()\n]{1,300}\)"),
    re.compile(r"\bconsole\.log\s*\([^()\n]{1,300}\)"),
    re.compile(r"\bSystem\.out\.println\s*\([^()\n]{1,300}\)"),
    re.compile(r"\b(?:def|class|function)\s+\w+[^\n]{0,250}"),
)


def _parse_rss_results(document: str) -> list[dict[str, str]]:
    root = ElementTree.fromstring(document)
    results = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        snippet = (item.findtext("description") or "").strip()
        if title and urlparse(url).scheme in {"http", "https"}:
            results.append(
                {
                    "title": unescape(title),
                    "url": url,
                    "snippet": unescape(snippet),
                }
            )
    return results


class _ResultsParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capture: str | None = None
        self.depth = 0

    def _finish_result(self) -> None:
        if self.current and self.current.get("title") and self.current.get("url"):
            self.results.append(self.current)
        self.current = None
        self.capture = None
        self.depth = 0

    def _destination(self, href: str) -> str:
        absolute = urljoin(self.base_url, href)
        parsed = urlparse(absolute)
        if parsed.hostname and (
            parsed.hostname == "duckduckgo.com"
            or parsed.hostname.endswith(".duckduckgo.com")
        ):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                absolute = target
        if urlparse(absolute).scheme not in {"http", "https"}:
            return ""
        return absolute

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and classes.intersection({"result__a", "result-link"}):
            self._finish_result()
            url = self._destination(attributes.get("href") or "")
            if url:
                self.current = {"title": "", "url": url, "snippet": ""}
                self.capture = "title"
                self.depth = 1
            return

        if self.current is not None and classes.intersection(
            {"result__snippet", "result-snippet"}
        ):
            self.capture = "snippet"
            self.depth = 1
        elif self.capture is not None:
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.capture is None:
            return
        self.depth -= 1
        if self.depth <= 0:
            self.capture = None
            self.depth = 0

    def handle_data(self, data: str) -> None:
        if self.current is not None and self.capture is not None:
            self.current[self.capture] += data


class _CodeBlockParser(HTMLParser):
    """Collect preformatted and code-tagged examples from a result page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] | None = None
        self.pre_depth = 0
        self.code_depth = 0

    def _finish_block(self) -> None:
        if self.current is not None:
            block = "".join(self.current).strip()
            if len(block) >= 8 and block not in self.blocks:
                self.blocks.append(block)
        self.current = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            if self.current is None:
                self.current = []
            self.pre_depth += 1
        elif tag == "code":
            if self.current is None:
                self.current = []
            self.code_depth += 1
        elif self.current is not None and tag in {"br", "div", "li", "p"}:
            self.current.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
            if self.pre_depth == 0:
                self._finish_block()
        elif tag == "code" and self.code_depth:
            self.code_depth -= 1
            if self.code_depth == 0 and self.pre_depth == 0:
                self._finish_block()
        elif self.current is not None and tag in {"div", "li", "p"}:
            self.current.append("\n")

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current.append(data)


def _is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    hostname = parsed.hostname.casefold()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        return ipaddress.ip_address(hostname).is_global
    except ValueError:
        return True


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_public_http_url(newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _extract_code_examples(query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
    if not CODE_REQUEST_RE.search(query):
        return []

    examples: list[dict[str, str]] = []
    for result in results[:2]:
        url = result["url"]
        if not _is_public_http_url(url):
            continue
        request = Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )
        try:
            opener = build_opener(_PublicRedirectHandler())
            with opener.open(request, timeout=8) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    continue
                charset = response.headers.get_content_charset() or "utf-8"
                document = response.read(MAX_SOURCE_BYTES).decode(charset, errors="replace")
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue

        parser = _CodeBlockParser()
        parser.feed(document)
        if parser.current is not None:
            parser._finish_block()
        for block in parser.blocks[:2]:
            examples.append({"title": result["title"], "url": url, "code": block[:6_000]})
            if len(examples) >= 3:
                return examples
    return examples


def _extract_snippet_examples(results: list[dict[str, str]]) -> list[dict[str, str]]:
    examples = []
    for result in results:
        for pattern in SNIPPET_CODE_PATTERNS:
            for match in pattern.finditer(result["snippet"]):
                code = match.group(1) if match.lastindex else match.group(0)
                code = " ".join(code.split()).strip()
                if code and all(example["code"] != code for example in examples):
                    examples.append(
                        {"title": result["title"], "url": result["url"], "code": code}
                    )
                if len(examples) >= 3:
                    return examples
    return examples


def search_web(query: str, limit: int = 5) -> str:
    """Return a few titled search results, snippets, and source URLs."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Search query must be {MAX_QUERY_LENGTH} characters or fewer.")

    failures: list[str] = []
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for provider, base_url, result_format in SEARCH_ENDPOINTS:
        params = {"q": query}
        if result_format == "rss":
            params["format"] = "rss"
        url = base_url + "?" + urlencode(params)
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=12) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                document = response.read(1_500_000).decode(charset, errors="replace")
        except HTTPError as error:
            failures.append(f"{provider} returned HTTP {error.code}")
            continue
        except URLError as error:
            failures.append(f"{provider} could not be reached: {error.reason}")
            continue

        if result_format == "rss":
            try:
                results = _parse_rss_results(document)
            except ElementTree.ParseError:
                results = []
        else:
            parser = _ResultsParser(base_url)
            parser.feed(document)
            parser._finish_result()
            results = parser.results
        if not results:
            failures.append(f"{provider} returned no readable result links")
            continue

        code_examples = _extract_code_examples(query, results)
        if not code_examples and CODE_REQUEST_RE.search(query):
            code_examples = _extract_snippet_examples(results)
        results = results[: max(1, min(limit, 5))]
        formatted = [f"Web search results for: {query}", f"Search source: {provider}"]
        if code_examples:
            formatted.append("Code examples extracted from result pages:")
            for example in code_examples:
                formatted.append(
                    f"From: {example['title']}\n"
                    f"URL: {example['url']}\n"
                    f"```\n{example['code']}\n```"
                )
        elif CODE_REQUEST_RE.search(query):
            formatted.append("No code blocks were extractable from the top result pages.")
        for index, result in enumerate(results, start=1):
            snippet = " ".join(result["snippet"].split())
            formatted.append(
                f"{index}. {result['title'].strip()}\n"
                f"URL: {result['url']}\n"
                f"Snippet: {snippet[:900] or '(no snippet provided)'}"
            )
        return "\n\n".join(formatted)

    raise RuntimeError("Web search failed: " + "; ".join(failures))
