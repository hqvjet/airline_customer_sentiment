from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
import time
import random
import pandas

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:197.0) Gecko/20100101 Firefox/197.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:196.0) Gecko/20100101 Firefox/196.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:195.0) Gecko/20100101 Firefox/195.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:194.0) Gecko/20100101 Firefox/194.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:193.0) Gecko/20100101 Firefox/193.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:192.0) Gecko/20100101 Firefox/192.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:191.0) Gecko/20100101 Firefox/191.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:190.0) Gecko/20100101 Firefox/190.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:189.0) Gecko/20100101 Firefox/189.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:188.0) Gecko/20100101 Firefox/188.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:187.0) Gecko/20100101 Firefox/187.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:186.0) Gecko/20100101 Firefox/186.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:185.0) Gecko/20100101 Firefox/185.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:184.0) Gecko/20100101 Firefox/184.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:183.0) Gecko/20100101 Firefox/183.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:182.0) Gecko/20100101 Firefox/182.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:181.0) Gecko/20100101 Firefox/181.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:180.0) Gecko/20100101 Firefox/180.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:179.0) Gecko/20100101 Firefox/179.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:178.0) Gecko/20100101 Firefox/178.0",
]

Bamboo = 'https://www.tripadvisor.com.vn/Airline_Review-d17550096-Reviews-Bamboo-Airways'
Vietjet = 'https://www.tripadvisor.com.vn/Airline_Review-d8728891-Reviews-VietJetAir'
VietnamA= 'https://www.tripadvisor.com.vn/Airline_Review-d8729180-Reviews-Vietnam-Airlines'
my_url = Vietjet

options = webdriver.FirefoxOptions()
options.add_argument(f"user-agent={random.choice(user_agents)}")
options.add_argument("--private")  # Sử dụng chế độ ẩn danh (private mode) của Firefox
driver = webdriver.Firefox(options=options)
driver.get(my_url)
wait = WebDriverWait(driver, 30)
time.sleep(10)

data = { 'Rating': [], 'Title': [], 'Content': []}

try:
    while True:
        time.sleep(random.uniform(5, 6))

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        div_elements = driver.find_elements(By.XPATH, "//div[@class='lszDU']")
        for div_element in div_elements:
            driver.execute_script("arguments[0].click();", div_element)
            time.sleep(random.uniform(0.5, 1))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Get DIV
        divs = soup.find_all('div', class_='WAllg _T')

        for div in divs:
            title_text = div.find('div', attrs={'data-test-target': 'review-title'}).find('a').find('span').find('span').text.strip()
            content_text = div.find('span', attrs={'data-test-target': 'review-text'}).find('span').text.strip()
            rated = div.find('div', attrs={'data-test-target': 'review-rating'}).find('span').get('class')[1]
            rated = int(rated.split('_')[1]) // 10
            rated = 1 if (rated == 1 or rated == 2) else (2 if rated == 3 else 3)
            print('TITLE', title_text)
            print('CONTENT', content_text)
            print('RATING', rated)
            data['Rating'].append(rated)
            data['Title'].append(title_text)
            data['Content'].append(content_text)
            print('-----------------------------------------------------------------------------')

        # next_button = wait.until(EC.visibility_of_element_located((By.XPATH, "//*[text()='Tiếp theo']")))
        next_button = driver.find_element(By.XPATH, "//*[text()='Tiếp theo']")
        if next_button.is_enabled:
            next_button.click()
        else:
            break

except Exception as e:
    print(e)

df = pandas.DataFrame(data)
df.to_csv('data/vietjet_en.csv', index=False)

driver.quit()
