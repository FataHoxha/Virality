
from __future__ import print_function
import httplib2
import os
import time

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

###
import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.client import AccessTokenRefreshError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.client import FlowExchangeError
###
from users_img_nav import Navigator
navigator = Navigator('/data0/pilzer/googleplus_dataset/users_img.txt')
navigator.loadUsers()
###
import urllib2
import re
###

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/people.googleapis.com-python-quickstart.json
SCOPES = [
    'https://www.googleapis.com/auth/contacts',
    'https://www.googleapis.com/auth/contacts.readonly',
    #'https://www.googleapis.con/auth/plus.login',
    'https://www.googleapis.com/auth/plus.me',
    'https://www.googleapis.com/auth/user.addresses.read',
    'https://www.googleapis.com/auth/user.birthday.read',
    'https://www.googleapis.com/auth/user.emails.read',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
]
CLIENT_SECRET_FILE = 'client_secrets.json'
APPLICATION_NAME = 'People API Python Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'people.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def save_activities(userId, credetials, service, person):
    count = 1
    try:
        activities_request = service.activities().list(userId=userId, collection='public', maxResults='50')
        #comments_request = service.comments()

        with open('/data0/pilzer/googleplus_dataset/activities_json/'+str(userId)+'.json','w') as file:
            activity_count = 0
            file.write('[')
            while activities_request != None:
                activities_document = activities_request.execute()
                if 'items' in activities_document:
                    print('+++++++\n+++++++++ got page with %d ++++++++++++++' % len(activities_document['items']))
                    print('+++++++ for person %d +++++++' % int(person))
                    for activity in activities_document['items']:
                        if 'attachments' in activity['object']:
                            attachments = activity['object']['attachments'][0]
                            if ('objectType' in attachments) and (attachments['objectType'] == 'photo') and ('displayName' in attachments) and ('image' in attachments):
                                if not attachments['image']['url'].endswith('.gif'):
                                    #print(activity['object']['attachments'][0]['image']['url'])
                                    parsed_activity = parse_activity(activity)
                                    parsed_activity["date"] = activity['published']
                                    print(person, count, activity['published'], activity['id'])
                                    parsed_activity["comments"] = retrieve_comments(service, activity['id'])
                                    count += 1
                                    if activity_count != 0:
                                        file.write(',\n')
                                    json.dump(parsed_activity, file, indent=4)
                                    activity_count = activity_count + 1
                                    print(len(parsed_activity["comments"]))
                            #attachments = activity['object']['attachments'][0]
                            #print(attachments['objectType'], activity['title'])
                            #print(attachments['image']['url'], attachments['content'])
                            #print(activity['object']['content'])
                            #print(attachments['image']['url'])
                        print('--------------')
                        #time.sleep(2)
                activities_request = service.activities().list_next(activities_request, activities_document)
            file.write(']')
            print('finished saving')
    except HttpError:
        with open('broken.txt','a') as file:
            file.write(str(userId)+str('\n'))
        print('Error: profile ' + str(userId))

def retrieve_comments(service, activity_id):
    comments_request = service.comments().list(maxResults=100, activityId=activity_id)
    parsed_comment = []
    while comments_request != None:
        comments_document = comments_request.execute()
        if 'items' in comments_document:
            print('got page with %d comments' % len( comments_document['items'] ))
            for comment in comments_document['items']:
                if comments_document['items'][0]['inReplyTo'][0]['id'] == activity_id:
                    parsed_comment.append(parse_comment(comment))
                    time.sleep(0.1)
        comments_request = service.activities().list_next(comments_request, comments_document)
    #print(parsed_comment)
    return parsed_comment

def parse_comment(comment):
    parsed_dict = {}
    parsed_dict["plusoners"] = comment['plusoners']['totalItems']
    parsed_dict["id"] = comment['id']
    parsed_dict["date"] = comment['published']
    parsed_dict["text"] = comment['object']['content']
    return parsed_dict

def parse_activity(activity):
    attachments = activity['object']['attachments'][0]
    parsed_dict = {}
    parsed_dict["text"] = attachments['displayName'] #text
    parsed_dict["imageUrl"] = attachments['image']['url'] #image url
    parsed_dict["reshares"] = activity['object']['resharers']['totalItems'] #resharers
    parsed_dict["plusoners"] = activity['object']['plusoners']['totalItems'] #plusoners
    parsed_dict["replies"] = activity['object']['replies']['totalItems'] #comments
    parsed_dict["id"] = attachments['id'] #activity identifier
    return parsed_dict

def followers(userId):
    try:
        user = urllib2.urlopen('https://plus.google.com/'+userId).read()
        re_exp = re.compile(r'<span class="BOfSxb">(.*?)\</span>', re.IGNORECASE|re.DOTALL)
        follower = re.findall(re_exp, user)
        print(userId, str(':  '), follower)
        return follower
    except:
        #print('Error: profile ' + str(userId))
        return 0



def main():
    """Shows basic usage of the Google People API.

    Creates a Google People API service object and outputs the name if
    available of 10 connections.
    """
    # crawl activities
    person = 1

    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = build('plus', 'v1', http=http)


    userId = navigator.nextUserId()
    while userId != 0:
        time.sleep(3)
        save_activities(userId, credentials, service, person)
        person += 1
        userId = navigator.nextUserId()
    """
    # retrieve followers
    #userId = navigator.nextUserId()
    userId1 = range(100000000000000652000,100000000000001000000)
    #while userId != 0:
    for userId in userId1:
        if userId % 1000 == 0:
            print(userId)
        f = followers(str(userId))
        #userId = navigator.nextUserId()
    """

if __name__ == '__main__':
    main()

