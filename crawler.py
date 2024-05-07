from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import cfscrape
import time
import csv
import re

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
num_pages = 50

names = []
links = []
epochs = []

def filter_models(model_data):
    filtered_models = []
    for model in model_data:
        # Loại bỏ dấu "
        model_name = model[0].replace('"', '')
        # Loại bỏ model có chứa chữ "SoVITS"
        if 'SoVITS' not in model_name:
            filtered_models.append(model)
    return filtered_models

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
        print(name)
        print(link)
        names.append(name)
        links.append(link)

    # Tìm nút chuyển trang và nhấp vào nó
    next_button = driver.find_element("css selector", 'li.page-item.next.m-1 a.page-link.px-0')
    next_button.click()
    time.sleep(1)

def filter_models(model_data):
    filtered_models = []
    for model in model_data:
        # Lấy tên model và loại bỏ dấu "
        model_name = model[0].strip('"')
        # Kiểm tra xem tên model có chứa chuỗi "RVC" hay không
        if 'RVC' in model_name:
            filtered_models.append(model)
    return filtered_models

# Lọc dữ liệu trước khi ghi vào file
filtered_data = filter_models(zip(names, links))

# Ghi dữ liệu đã lọc vào file CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Link"])
    for name, link in filtered_data:
        writer.writerow([name, link])
