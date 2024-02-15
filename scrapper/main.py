################################################################################
### Step 1
################################################################################

import re
import time
from datetime import datetime, timedelta
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define OpenAI api_key
# openai.api_key = '<Your API Key>'

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
### Step 2
################################################################################

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
### Step 3
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


################################################################################
### Step 4
################################################################################

def tmp_write_file(text):
    nanoseconds_timestamp = str(int(time.time() * 1e9))

    temp_filename = f"/tmp/crawled_data_{nanoseconds_timestamp}.txt"
    with open(temp_filename, 'w', encoding='utf-8') as file:
        file.write(text)

    return temp_filename

def crawl(url, site_map=False):
    started_time = datetime.now()

    crawled_links = []
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # While the queue is not empty, continue crawling
    while queue:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')  # Disable GPU for headless mode
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                    
        # Create a Selenium webdriver with headless options
        driver = webdriver.Chrome(options=chrome_options) 

        url = queue.pop()
        
        driver.get(url)

        driver.implicitly_wait(4)  # Wait for 2 seconds

        # Get the page source, which includes the dynamically loaded content
        page_source = driver.page_source
        
        
        # Try extracting the text from the link, if failed proceed with the next item in the queue
        file_name, text, text_length = None, None, 0
        try:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(page_source, "html.parser")
            links_text = "" 
            for link in soup.find_all("a"):
                links_text += link.get("href") + " : " + link.text + "\n"

            # Get the text but remove the tags
            text = soup.get_text() 
            text += links_text

            text_length = len(text)
            text = re.sub(r'\n\s*\n', '\n', text)
            file_name = tmp_write_file(text)

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")
            
        except Exception as e:
            print("Unable to parse page " + url, "Error:", e)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

        if text_length == 0:
            continue
        crawled_links.append({"url":url, "file_name":file_name, "text_length":text_length})

        driver.quit()

        if datetime.now()-timedelta(minutes=3) > started_time or site_map: # only have to scrap for 2 minutes to get better performance
            break
        
    return crawled_links