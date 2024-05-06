from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import cfscrape
import time
import csv
import re


def get_epochs(name):
    # Biểu thức chính quy để tìm số epochs trong tên model
    pattern = r'(\d+)\s*(?:epoch|epochs|e)'

    # Tìm kiếm tất cả các lần xuất hiện của mẫu trong tên model
    matches = re.findall(pattern, name, re.IGNORECASE)

    if matches:
        # Lấy số epochs lớn nhất trong tên
        epochs = max(map(int, matches))
        return epochs
    else:
        return None


session = cfscrape.create_scraper()
cookies = session.cookies.get_dict()

# Khởi tạo trình duyệt
driver = webdriver.Chrome()

# Mở trang web
driver.get('https://voice-models.com/')

# Thêm cookies vào trình duyệt
for cookie_name, cookie_value in cookies.items():
    driver.add_cookie({'name': cookie_name, 'value': cookie_value})
driver.refresh()

# Số trang bạn muốn crawl
num_pages = 10

names = []
links = []
epochs = []

for i in range(num_pages):
    # Chờ cho đến khi dữ liệu AJAX được tải
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'tableBody'))
    )

    # Parse dữ liệu HTML bằng BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table_body = soup.find('tbody', {'id': 'tableBody'})
    rows = table_body.find_all('tr')

    for row in rows:
        columns = row.find_all('td')
        name = columns[0].find('a').text
        link = columns[2].find('a')['href']
        epoch = get_epochs(name)
        print(name)
        print(link)
        print(epoch)
        names.append(name)
        links.append(link)
        epochs.append(epoch)

    # Tìm nút chuyển trang và nhấp vào nó
    next_button = driver.find_element("css selector", 'li.page-item.next.m-1 a.page-link.px-0')
    next_button.click()
    time.sleep(1)

# Ghi dữ liệu vào file CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Link", "Epochs"])
    for name, link, epoch in zip(names, links, epochs):
        writer.writerow([name, link, epoch])