import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
import os
import logging
import sys

logger = logging.getLogger('crawler')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

handbook_url = 'https://www.handbook.fca.org.uk/handbook/PROD/'
handbook_directory = "./handbook"

async def save_page(directory, url, content):
    if not url.endswith('.pdf') and not url.endswith('html'):
        return

    try:
        filename = url.replace('https://', '').replace('/', '-')
        path = os.path.join(directory, filename)
        with open(path, 'wb') as file:
            file.write(content)
    except Exception as e:
        logger.error(f"Error saving the file content: {url}")


async def parse_page(url, content, queue):
    if url.endswith('.pdf'):
        return

        # parse the page content
    soup = BeautifulSoup(content, 'html.parser')

    # find all the links on the page
    links = soup.find_all('a')

    for link in links:
        href = link.get('href')
        if href is None or not href.startswith('/handbook/'):
            continue

        href = urljoin(url, href)
        await queue.put(href)


async def visit_page(session, url, visited, queue, semaphore):
    async with semaphore:
        url, _ = urldefrag(url)
        url = urlunparse(urlparse(url)._replace(query=""))

        # prevent revisiting the same page
        if url in visited:
            return

        # mark the page as visited
        visited.add(url)

        logger.info(f"Visiting {url}")
        async with session.get(url) as response:
            content = await response.content.read()
            await save_page(handbook_directory, url, content)
            await parse_page(url, content, queue)


async def main(concurrency):
    visited = set()
    queue = asyncio.Queue()
    await queue.put(handbook_url)

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while queue:
            tasks = []
            url = await queue.get()
            task = asyncio.ensure_future(visit_page(session, url, visited, queue, semaphore))
            tasks.append(task)
            await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main(20))
