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

def get_epochs(name):
    # Biểu thức chính quy để tìm số epochs trong tên model
    pattern = r'(\d+)\s*(?: epoch epochs | e)'

    # Tìm kiếm tất cả các lần xuất hiện của mẫu trong tên model
    matches = re.findall(pattern, name, re.IGNORECASE)

    if matches:
        # Lấy số epochs lớn nhất trong tên
        epochs = max(map(int, matches))
        return epochs
    else:
        return None


def filter_models(model_data):
    filtered_models = []
    for model in model_data:
        # Lấy tên model và loại bỏ dấu "
        model_name = model[0].strip('"')
        # Lấy số epochs từ tên model
        epochs = get_epochs(model_name)

        # Kiểm tra điều kiện lọc
        if epochs is not None and epochs >= 100 and (
                'SoVITS' in model_name or 'RVC' in model_name or 'OV2' in model_name or 'KLMv7sE' in model_name or 'RWBY' in model_name):
            filtered_models.append(model)
    return filtered_models

# Lọc dữ liệu trước khi ghi vào file
filtered_data = filter_models(zip(names, links))

# Đếm số lượng model
total_models = len(names)
print(f"Tổng số model crawl được: {total_models}")

# Đếm số lượng model không đúng kiểu
invalid_models = [name for name in names if not any(keyword in name for keyword in ['SoVITS', 'RVC', 'OV2', 'KLMv7sE', 'RWBY'])]
num_invalid_models = len(invalid_models)
print(f"Số model không đúng kiểu: {num_invalid_models}")

# Đếm số lượng model có epoch thấp
low_epoch_models = [name for name, _ in zip(names, links) if get_epochs(name) is not None and get_epochs(name) < 100]
num_low_epoch_models = len(low_epoch_models)
print(f"Số model có epoch thấp: {num_low_epoch_models}")

# Đếm số lượng model có thể sử dụng
usable_models = len(filtered_data)

print(f"Số model có thể sử dụng: {usable_models}")
# Ghi dữ liệu đã lọc vào file CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Link"])
    for name, link in filtered_data:
        writer.writerow([name, link])
