from bs4 import BeautifulSoup
import requests
import pandas as pd
import os

headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
    "cache-control": "max-age=0",
    "cookie": "__cfduid=d1400b10fdb7a343e47a28b7ae14d26101587836657; s_fid=102501B0099E1AA3-2F1873F3B42B276F; s_vi=[CS]v1|2F523B7A05158044-40000945055A7307[CE]; cf_clearance=88907b123ae15814240fd9f2a1abce35eb18d4e3-1587869143-0-150; PHPSESSID=5prauina15lsmegdakb99tdgfh; searchResultViewType%2Fsearch=expanded; KWICView%2Fsearch=false; s_cc=true",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"
}

# 从个人主页上面获取twitter和facebook账号
def find_twitter_facebook(url):
    try:
        f = requests.get(url,headers=headers)
        soup = BeautifulSoup(f.content, "html.parser")  # 用lxml解析器解析该网页的内容, 好像f.text也是返回的html
        twitter_account = ''
        for tag in soup.find_all(True):
           if ('href' in tag.attrs) and ('twitter.com' in tag['href']) and tag['href'] != twitter_account \
                   and (len(tag['href']) <= len(twitter_account) or len(twitter_account) == 0):
               twitter_account = tag['href']
    except:
        twitter_account = ''
    try:
        facebook_account = ''
        for tag in soup.find_all(True):
            if ('href' in tag.attrs) and '//www.facebook.com' in tag['href'] and tag['href'] != facebook_account \
                   and (len(tag['href']) <= len(facebook_account) or len(facebook_account) == 0):
               facebook_account = tag['href']
    except:
        facebook_account = ''
    return twitter_account,facebook_account


# 获取所有的议员列表页面（默认6页）的url
def get_all_rep_list(url):
    url_rep_list = {}
    url_rep_list["1"] = url  # 起始页
    changed = True
    while (changed):
        changed = False
        f_rep_list = requests.get(list(url_rep_list.values())[-1],headers=headers, timeout=20)
        soup = BeautifulSoup(f_rep_list.content, "html.parser")
        for tag in soup.find_all('div', 'nav-pag-top', 'navigation'):
            for temp in tag.find_all('a'):
                if temp.get_attribute_list('class')[0] == None and temp.text not in url_rep_list:
                    url_rep_list[temp.text] = "https://www.congress.gov"+temp.get_attribute_list('href')[0]
                    changed = True
    return list(url_rep_list.values())

# 在某页议员列表里面获取议员名字和议员个人简介页面的地址
def find_name_intro_web(url):
    f_rep_list = requests.get(url,headers=headers)
    soup = BeautifulSoup(f_rep_list.content, "html.parser")
    # 当页的议员名字列表
    name_list = list()
    # 当页议员对应的个人简介网站
    personal_intro_web = list()
    for tag in soup.find_all('li', 'expanded'):
        temp = tag.find_all('span', "result-heading")
        for i in temp:
            name_list.append(i.string)
            personal_intro_web.append(i.a.get_attribute_list('href')[0])
    return name_list, personal_intro_web


# 从个人简介页面上面获取每个人的个人主页
def find_personal_web(url):
    try:
        f_intro_web = requests.get(url,headers=headers)
        soup_intro_web = BeautifulSoup(f_intro_web.content, "html.parser")
        temp_intro = soup_intro_web.find_all('table', 'standard01 nomargin')[0]
        personal = temp_intro.find_all('a')
        if personal != None:
            personal = personal[0].get_attribute_list('href')[0]
        else:
            personal = ''
        return personal
    except:
        return ''


if __name__ == '__main__':
    #ini_url = "https://www.congress.gov/search?searchResultViewType=expanded&KWICView=false&q=%7B%22source%22%3A%22members%22%2C%22congress%22%3A%5B%22115%22%5D%7D&pageSize=100&page=1"  #限制了任期为2017-2018的议员
    #ini_url='https://www.congress.gov/search?searchResultViewType=expanded&KWICView=false&q=%7B%22source%22%3A%22members%22%2C%22congress%22%3A%22114%22%7D&pageSize=100&page=1'
    ini_url='https://www.congress.gov/search?searchResultViewType=expanded&q=%7B%22source%22%3A%22members%22%2C%22congress%22%3A116%7D'
    all_rep_list_page = get_all_rep_list(ini_url)
    rep_names = list()
    intro_web = list()
    personal_web = list()
    twitter_web = list()
    facebook_web = list()
    i=0
    for rep_page in all_rep_list_page:
        temp_name, temp_page = find_name_intro_web(rep_page)
        rep_names = rep_names + temp_name
        intro_web = intro_web + temp_page
        i = i+1
        print('rep %f' %(i/len(all_rep_list_page)))
    i = 0
    for intro in intro_web:
        personal_web.append(find_personal_web(intro))
        i = i+1
        print('personal %f' % (i / len(intro_web)))
    i = 0
    for personal in personal_web:
        temp_twitter, temp_facebook = find_twitter_facebook(personal)
        twitter_web.append(temp_twitter)
        facebook_web.append(temp_facebook)
        i = i+1
        print('twitter facebook %f' % (i / len(personal_web)))
    output = pd.DataFrame(
        {'Name': rep_names, 'Intro_web': intro_web, 'Personal_web': personal_web, 'Twitter': twitter_web,
         'Facebook': facebook_web})
    output.to_csv('1920output.csv',sep=',')
