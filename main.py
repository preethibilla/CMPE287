# This is a sample Python script.
import matplotlib.pyplot as plt
import pytest
import numpy as np
import time
import json
import csv
import spacy
import undetected_chromedriver.v2 as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

def pieplot(pos,neg,title,fname):
    fig1, ax1 = plt.subplots()
    ax1.pie([(pos/(pos+neg))*100,(neg/(pos+neg))*100], labels=['Passed','Failed'], autopct='%1.1f%%',shadow=True, startangle=90,colors=['#23C552','#F84F31'])
    ax1.axis('equal')
    ax1.set_title(title)
    plt.savefig(fname)

def barplot(labels, poslist,neglist,title):
    X_axis = np.arange(len(labels))
    #fig,ax = plt.subplots()
    plt.bar(X_axis - 0.2,poslist,0.4,label="Passed",color='g')
    plt.bar(X_axis + 0.2,neglist,0.4,label="Failed",color='r')
    plt.xticks(X_axis, labels)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("AI testing Summary")
    plt.legend()
    plt.savefig(title)


class TestScriptsample():
    def __init__(self):
        self.driver = uc.Chrome()
        self.positive = 0
        self.negative = 0
        self.start_time = time.time()

    def read_data(self, filename):
        with open(filename,encoding='mac_roman') as csvfile:
            reader = csv.reader(csvfile)
            self.data = list(reader)[1:]
    def preprocess_word(self, word):
        word = word.lower()
        res = ''
        for s in word:
            if s.isalpha() or s ==' ' or s.isnumeric():
                res += s
        return res

    def test_scriptsample(self):
        self.driver.get("https://chat.kuki.ai/chat")
        self.driver.set_window_size(1258, 1040)
        element = self.driver.find_element(By.CSS_SELECTOR, ".g-sign-in-button .text-container > span")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click_and_hold().perform()
        element = self.driver.find_element(By.CSS_SELECTOR, ".g-sign-in-button > .content-wrapper")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).release().perform()
        self.driver.find_element(By.CSS_SELECTOR, ".g-sign-in-button > .content-wrapper").click()
        self.driver.implicitly_wait(5)
        loginBox = self.driver.find_element(by=By.XPATH, value='//*[@id ="identifierId"]')
        loginBox.send_keys("kuki.masapeta@gmail.com")
        nextButton = self.driver.find_elements(by=By.XPATH, value='//*[@id ="identifierNext"]')
        nextButton[0].click()
        self.driver.implicitly_wait(7)
        try:
            passWordBox = self.driver.find_element(by=By.XPATH, value='//*[@id ="password"]/div[1]/div / div[1]/input')
        except:
            self.driver.implicitly_wait(7)
            passWordBox = self.driver.find_element(by=By.XPATH, value='//*[@id ="password"]/div[1]/div / div[1]/input')
        passWordBox.send_keys("Kuki@1997")
        nextButton = self.driver.find_elements(by=By.XPATH, value='//*[@id ="passwordNext"]')
        nextButton[0].click()
        nlp = spacy.load('en_core_web_sm')
        cid = "pb-bot-response"
        time.sleep(5)
        for i in range(len(self.data)):
            try:
                in_message = self.data[i][0]
                out_messages = self.data[i][1]
                out_messages = out_messages.split(",")
                time.sleep(5)
                self.driver.find_element(By.CSS_SELECTOR, "input").send_keys(in_message)
                self.driver.find_element(By.CSS_SELECTOR, "input").send_keys(Keys.ENTER)
                time.sleep(2)
                ids = self.driver.find_elements(By.CLASS_NAME, cid)
                output = ids[-1].text
                output = self.preprocess_word(output)
                real_out = nlp(output)
                similar = 0
                for out_message in out_messages:
                    wish_out = nlp(out_message)
                    simi2 = real_out.similarity(wish_out)
                    similar = max(similar, simi2)

                print("Test case %s:" % str(i))
                print("Input: %s" % in_message)
                print("Output: %s" % output)
                if similar >0.30:
                    self.positive += 1
                    print("True")
                else:
                    self.negative += 1
                    print("False")
                print()
            except:
                continue


if __name__ == '__main__':

    dk_sample = TestScriptsample()
    dk_sample.read_data("domain_knowledge1.csv")
    dk_sample.test_scriptsample()
    end_time = time.time()
    print("Total positive results for Domain Knowledge Testing: %d" % dk_sample.positive)
    print("Total negative results for Domain Knowledge Testing: %d" % dk_sample.negative)
    print("Total time consuming for Domain Knowledge Testing: %d" % (end_time-dk_sample.start_time))
    dk_positives = dk_sample.positive
    dk_negatives = dk_sample.negative
    pieplot(dk_positives,dk_negatives,"PieChart for Domain Knowledge Testing","DK_PieChart.png")


    cm_sample = TestScriptsample()
    cm_sample.read_data("chat_memory1.csv")
    cm_sample.test_scriptsample()
    end_time = time.time()
    print("Total positive results for Chat Memory Testing: %d" % cm_sample.positive)
    print("Total negative results for Chat Memory Testing: %d" % cm_sample.negative)
    print("Total time consuming for Chat Memory Testing: %d" % (end_time-cm_sample.start_time))
    cm_positives = cm_sample.positive
    cm_negatives = cm_sample.negative
    pieplot(cm_positives,cm_negatives,"PieChart for Chat Memory Testing","CM_PieChart.png")


    cp_sample = TestScriptsample()
    cp_sample.read_data("chat_pattern1.csv")
    cp_sample.test_scriptsample()
    end_time = time.time()
    print("Total positive results for Chat Pattern Testing: %d" % cp_sample.positive)
    print("Total negative results for Chat Pattern Testing: %d" % cp_sample.negative)
    print("Total time consuming for Chat Pattern Testing: %d" % (end_time-cp_sample.start_time))
    cp_positives = cp_sample.positive
    cp_negatives = cp_sample.negative
    pieplot(cp_positives,cp_negatives,"PieChart for Chat Pattern Testing","CP_PieChart.png")


    qa_sample = TestScriptsample()
    qa_sample.read_data("qa_testing1.csv")
    qa_sample.test_scriptsample()
    end_time = time.time()
    print("Total positive results for QA Testing: %d" % qa_sample.positive)
    print("Total negative results for QA Testing: %d" % qa_sample.negative)
    print("Total time consuming for QA Testing: %d" % (end_time-qa_sample.start_time))
    qa_positives = qa_sample.positive
    qa_negatives = qa_sample.negative
    pieplot(qa_positives,qa_negatives,"PieChart for Q&A Testing","QA_PieChart.png")

    labels = ['Domain Knowledge', 'Chat Memory', 'Chat Pattern', 'QA_Testing']
    poslist = [dk_positives,cm_positives,cp_positives,qa_positives]
    neglist = [dk_negatives,cm_negatives,cp_negatives,qa_negatives]
    barplot(labels,poslist,neglist,"AI Testing Summary.png")





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
