from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import random
import pandas as pd
import traceback

def crawlData():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:297.0) Gecko/20100101 Firefox/197.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:296.0) Gecko/20100101 Firefox/196.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:295.0) Gecko/20100101 Firefox/195.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:294.0) Gecko/20100101 Firefox/194.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:293.0) Gecko/20100101 Firefox/193.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:292.0) Gecko/20100101 Firefox/192.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:291.0) Gecko/20100101 Firefox/191.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:290.0) Gecko/20100101 Firefox/190.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:289.0) Gecko/20100101 Firefox/189.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:288.0) Gecko/20100101 Firefox/188.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:287.0) Gecko/20100101 Firefox/187.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:286.0) Gecko/20100101 Firefox/186.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:285.0) Gecko/20100101 Firefox/185.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:284.0) Gecko/20100101 Firefox/184.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:283.0) Gecko/20100101 Firefox/183.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:282.0) Gecko/20100101 Firefox/182.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:281.0) Gecko/20100101 Firefox/181.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:280.0) Gecko/20100101 Firefox/180.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:279.0) Gecko/20100101 Firefox/179.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:278.0) Gecko/20100101 Firefox/178.0",
    ]

    web = 'https://www.tripadvisor.com.vn/Airlines'

    options = webdriver.FirefoxOptions()
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    options.add_argument("--private")  # Sử dụng chế độ ẩn danh (private mode) của Firefox
    driver = webdriver.Firefox(options=options)
    driver.set_window_size(1280, 1024)
    driver.get(web)

    data = {'Rating': [], 'Title': [], 'Content': []}
    page = 1
    def scroll_to_element(element):
        driver.execute_script("arguments[0].scrollIntoView();", element)
        time.sleep(0.2)

    time.sleep(10)

    try:
        while True:

            time.sleep(random.uniform(2, 3))
            groups = WebDriverWait(driver, 50).until(
                EC.presence_of_all_elements_located((By.XPATH, ".//div[@class='cell left']"))  
            )

            for group in groups:
                print('PAGE: ', page)
                link = group.find_element(By.XPATH, ".//a[@class='detailsLink']")
                scroll_to_element(link)
                actions = ActionChains(driver)
                actions.key_down(Keys.CONTROL).click(link).key_up(Keys.CONTROL).perform()

                driver.switch_to.window(driver.window_handles[1])
                WebDriverWait(driver, 50).until(lambda d: d.execute_script('return document.readyState') == 'complete')
                try:
                    more = WebDriverWait(driver, 50).until(
                        EC.presence_of_element_located((By.XPATH, "//div[@class='EfbKZ b _S']"))
                    ) 
                    scroll_to_element(more)
                    driver.execute_script("arguments[0].click();", more)
                except:
                    print('NO MORE EXTRA LANGUAGES!')
                time.sleep(1)
                try:
                    checker = WebDriverWait(driver, 50).until(
                        EC.presence_of_element_located((By.XPATH, "//input[@value='en']"))
                    ) 
                    scroll_to_element(checker)
                    driver.execute_script("arguments[0].click();", checker)
                except:
                    print('NO ENGLISH LANGUAGE!')
                    continue
                time.sleep(1)

                while True:
                    
                    # time.sleep(random.uniform(3, 4))
                    time.sleep(2)
                    try:
                        comments = WebDriverWait(driver, 50).until(
                            EC.presence_of_all_elements_located((By.XPATH, ".//div[@class='WAllg _T']"))
                        )
                    except:
                        break

                    print('COMMENTS NUM: ', len(comments))
                    WebDriverWait(driver, 50).until(
                        EC.presence_of_all_elements_located((By.XPATH, ".//span[contains(@class, 'ui_bubble_rating')]"))
                    )
                    WebDriverWait(driver, 50).until(
                        EC.presence_of_all_elements_located((By.XPATH, ".//button[contains(@class, 'ui_button secondary small')]"))
                    )
                    # expand = driver.find_elements(By.XPATH, ".//div[contains(@class, 'lszDU')]")[0]
                    # scroll_to_element(expand)
                    # driver.execute_script("arguments[0].click();", expand)
                    if len(comments) > 0:
                        for comment in comments:
                            scroll_to_element(comment)

                            rating = comment.find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating')]")
                            rating = int(rating.get_attribute('class').split()[1].split('_')[1])
                            rating = 1 if rating < 30 else 2 if rating == 30 else 3

                            translator = comment.find_element(By.XPATH, ".//button[contains(@class, 'ui_button secondary small')]")
                            driver.execute_script("arguments[0].click();", translator)
                            # time.sleep(random.uniform(2, 2.5))

                            title_element = WebDriverWait(driver, 50).until(
                                EC.presence_of_element_located((By.XPATH, "//div[@class='quote']"))
                            )
                            title = title_element.text.strip()

                            content_element = WebDriverWait(driver, 50).until(
                                EC.presence_of_element_located((By.XPATH, "//div[@class='entry']"))  
                            )
                            content = content_element.text.strip()

                            print('RATING: ', rating)
                            print('TITLE: ', title)
                            print('CONTENT: ', content)
                            data['Rating'].append(rating)
                            data['Title'].append(title)
                            data['Content'].append(content)
                            print('------------------------------------------------')

                            close = driver.find_element(By.XPATH, "//div[@class='zPIck _Q Z1 t _U c _S zXWgK']")
                            driver.execute_script("arguments[0].click();", close)

                    next_button = driver.find_elements(By.XPATH, "//a[@class='ui_button nav next primary ']")
                    if len(next_button) > 0:
                        scroll_to_element(next_button[0])
                        driver.execute_script("arguments[0].click();", next_button[0])
                        print('HAS PAGINATION')
                    else:
                        print('PAGINATION END')
                        break
                driver.close()
                driver.switch_to.window(driver.window_handles[0])

            print('FINISH THIS PAGE')
            next_btn = driver.find_element(By.XPATH, "//span[@class='nav next ui_button primary']")
            scroll_to_element(next_btn)
            disabled = next_btn.get_attribute('class').split()[-1]
            if disabled == 'disabled':
                print('STOPPED !!!!')
                break
            else:
                driver.execute_script("arguments[0].click();", next_btn)
                page += 1
        driver.close()
    except Exception as e:
        print(e)
        traceback.print_exc()

    df = pd.DataFrame(data)
    df.to_csv('data/data_en1.csv', index=False)
    driver.quit()

crawlData()

print('ALL CRAWLER DONE!!')
