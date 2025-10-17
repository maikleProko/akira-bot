from time import sleep

from selenium.common import NoSuchElementException
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ADDRESS = "127.0.0.1:9222"

class SeleniumProcessor:
    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("debuggerAddress", ADDRESS)
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 60)
        self.actions = ActionChains(self.driver)
        self.run()

    def get_inner_element(self, parent, attribute_name, attribute_value):
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((attribute_name, attribute_value)))
        element = parent.find_element(attribute_name, attribute_value)
        try:
            logger.info("Element with attribute {} value {} is displayed".format(attribute_name, attribute_value))
        except AssertionError:
            pass
        return element

    def get_element(self, attribute_name, attribute_value):
        return self.get_inner_element(self.driver, attribute_name, attribute_value)

    def get_span_with_value(self, value):
        span_elements = self.driver.find_elements(By.TAG_NAME, 'span')
        for span in span_elements:
            if span.text == str(value):
                return span
        return None


    def click(self, type_name, type_value):
        element = self.get_element(type_name, type_value)
        WebDriverWait(self.driver, 600).until(
            EC.element_to_be_clickable((type_name, type_value))
        )
        element.click()
        logger.info("Element with css_selector {} is clicked".format(type_value))

    def click_selector(self, css_selector):
        element = self.get_element(By.CSS_SELECTOR, css_selector)
        WebDriverWait(self.driver, 600).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector))
        )
        element.click()
        logger.info("Element with css_selector {} is clicked".format(css_selector))

    def click_class(self, class_name):
        element = self.get_element(By.CLASS_NAME, class_name)
        WebDriverWait(self.driver, 600).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, class_name))
        )
        element.click()
        logger.info("Element with css_selector {} is clicked".format(class_name))

    def click_span_with_value(self, value):
        self.get_span_with_value(value).click()
        logger.info("Element which text is {} is clicked".format(value))

    def write(self, attribute_name, attribute_value, value):
        self.get_element(attribute_name, attribute_value).send_keys(value)
        logger.info("Was written {} on element with css_celector {}".format(value, attribute_value))

    def write_xpath(self, xpath, value):
        self.get_element(By.XPATH, xpath).send_keys(value)
        logger.info("Was written {} on element with xpath {}".format(value, xpath))

    def click_xpath(self, xpath):
        element = self.get_element(By.XPATH, xpath)
        WebDriverWait(self.driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        logger.info("Element with xpath {} is clicked".format(xpath))

    def close(self, substring):
        for handle in self.driver.window_handles:
            self.driver.switch_to.window(handle)
            if substring in self.driver.title:
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[-1])

    def go(self, url):
        previous_url = self.driver.current_url
        self.driver.get(url)
        WebDriverWait(self.driver, timeout=300).until(EC.url_changes(previous_url))
        logger.info("Go to link {}".format(url))

    def go_no_wait(self, url):
        self.driver.get(url)
        sleep(1)
        logger.info("Go to link {}".format(url))

    def go_no_check(self, url):
        previous_url = self.driver.current_url
        self.driver.get(url)
        sleep(5)
        logger.info("Go to link {}".format(url))

    def go_with_wait(self, url):
        self.driver.get(url)
        logger.info("Go to link {}".format(url))
        WebDriverWait(self.driver, timeout=30).until(
            lambda driver: driver.execute_script("""
                return document.readyState === 'complete' &&
                       typeof window.jQuery !== 'undefined' &&
                       jQuery.active === 0 &&
                       document.querySelectorAll('.dynamic-content').length > 0
            """)
        )
        logger.info("Gan to link {}".format(url))

    def run_realtime(self):
        pass

    def hover(self, css_selector):
        element = self.get_element(By.CSS_SELECTOR, css_selector)
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()
        logger.info("Element with css_selector {} is hovered".format(css_selector))

    def double_click_selector(self, css_selector):
        element = self.get_element(By.CSS_SELECTOR, css_selector)
        actions = ActionChains(self.driver)
        actions.move_to_element(element)
        actions.double_click(element).perform()
        logger.info("Element with css_selector {} is double-clicked".format(css_selector))

    def double_click_class(self, class_name):
        element = self.get_element(By.CLASS_NAME, class_name)
        actions = ActionChains(self.driver)
        actions.move_to_element(element)
        actions.double_click(element).perform()
        logger.info("Element with class {} is double-clicked".format(class_name))

    def double_click_xpath(self, xpath):
        element = self.get_element(By.XPATH, xpath)
        actions = ActionChains(self.driver)
        actions.move_to_element(element)
        actions.double_click(element).perform()
        logger.info("Element with xpath {} is double-clicked".format(xpath))

    def get_parent_element(self, element):
        return element.find_element(By.XPATH, "..")

    def find_element_by_text(self, parent_element, text):
        try:
            element = parent_element.find_element(
                By.XPATH, ".//*[text()='{}']".format(text)
            )
            return element
        except NoSuchElementException:
            return None

    def get_common_near_parent_element(self, element, css_selector):
        parent = element.find_element('xpath', './parent::*')
        try:
            child = self.get_inner_element(parent, By.CSS_SELECTOR, css_selector)
            return child
        except:
            return self.get_common_near_parent_element(parent, css_selector) if parent else None

    def get_element_using_text(self, text):
        xpath = f"//*[contains(text(), '{text}')]"
        wait = WebDriverWait(self.driver, 600)
        wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        element = self.driver.find_element(By.XPATH, xpath)
        return element

    def get_element_in_element_using_text(self, element, text):
        xpath = f".//*[contains(text(), '{text}')]"
        wait = WebDriverWait(self.driver, 600)
        wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        return element.find_element(By.XPATH, xpath)

    def get_element_using_near_text(self, text, css_selector):
        text_element = self.get_element_using_text(text)
        css_selector_element = self.get_common_near_parent_element(text_element, css_selector)
        return css_selector_element

    def click_using_near_text(self, text, css_selector):
        self.get_element_using_near_text(text, css_selector).click()
        logger.info("Element which label {} and css_selector {} is clicked".format(text, css_selector))

    def click_using_text(self, text):
        self.get_element_using_text(text).click()
        logger.info("Element which text {} is clicked".format(text))

    def double_click_using_near_text(self, text, css_selector):
        element = self.get_element_using_near_text(text, css_selector)
        actions = ActionChains(self.driver)
        actions.move_to_element(element)
        actions.double_click(element).perform()
        logger.info("Element which label {} and css_selector {} is double-clicked".format(text, css_selector))

    def write_using_near_text(self, text, css_selector, value):
        self.get_element_using_near_text(text, css_selector).send_keys(value)
        logger.info("Was written {} on element with label {} and css_selector {}".format(value, text, css_selector))

    def finish(self):
        sleep(2)
        logger.info('Finishing selenium test\n')
        self.driver.close()

    def send_key_combination_element(self, type_name, type_value, modifier_key, regular_key):
        element = self.get_element(type_name, type_value)
        actions = ActionChains(self.driver)

        if element:
            actions.move_to_element(element)

        actions.key_down(modifier_key)
        actions.send_keys(regular_key)
        actions.key_up(modifier_key)

        actions.perform()
        logger.info("Element {} is sended by key_combination".format(element))

    def tap(self, key):
        sleep(1)
        actions = ActionChains(self.driver)
        actions.send_keys(key)
        actions.perform()
        sleep(2)

    def tap_down_element(self, type_name, type_value):
        sleep(1)
        self.driver.find_element(type_name, type_value).send_keys(Keys.ARROW_DOWN)
        sleep(1)

    def screen(self, type_name, type_value, filename: str, offset: int = 0) -> bool:
        try:
            element = self.get_element(type_name, type_value)
            size = element.size
            element.screenshot(filename)

            from PIL import Image
            img = Image.open(filename)
            cropped_img = img.crop((offset, 0, size['width'], size['height']))
            cropped_img.save(filename)

            logger.info("Element {} is screened".format(element))
            return True
        except Exception as e:
            logger.error(e)
            return False

    def get_screen(self, type_name, type_value, offset: int = 0):
        try:
            element = self.get_element(type_name, type_value)
            size = element.size
            element.screenshot('files/tmp.png')

            from PIL import Image
            img = Image.open('files/tmp.png')
            logger.info("Element {} is screened".format(element))
            return img.crop((offset, 0, size['width'], size['height']))
        except Exception as e:
            logger.error(e)
            return None
