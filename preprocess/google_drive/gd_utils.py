import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive']


def get_and_save_access():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            credentials = flow.run_local_server(port=8070)
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
    return credentials


def find_folders_with_substring(service, substring, logger):
    page_token = None
    while True:
        response = service\
            .files()\
            .list(q="mimeType='application/vnd.google-apps.folder' and fullText contains '{}'".format(substring),
                  spaces='drive',
                  fields='nextPageToken, files(id, name)',
                  pageToken=page_token)\
            .execute()
        for file in response.get('files', []):
            logger.info('Found file: %s (%s)' % (file.get('name'), file.get('id')))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

# deletes(puts in trash)
# def delete_by_id(f_id):
#     r = service \
#         .files()\
#         .delete(fileId=f_id)\
#         .execute()


def get_folder_id(service, name, parent_id=None):
    page_token = None
    while True:
        response = service \
            .files() \
            .list(q="mimeType='application/vnd.google-apps.folder' and name = '{}'".format(name),
                  spaces='drive',
                  fields='nextPageToken, files(id, name, parents)',
                  pageToken=page_token).execute()
        for result in response.get('files', []):
            if parent_id is None:
                return result.get('id')
            else:
                if parent_id in result.get('parents'):
                    return result.get('id')
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return None


def get_file_id(service, name, parent_id=None):
    page_token = None
    while True:
        response = service \
            .files() \
            .list(q="mimeType!='application/vnd.google-apps.folder' and name = '{}'".format(name),
                  spaces='drive',
                  fields='nextPageToken, files(id, name, parents)',
                  pageToken=page_token).execute()
        for result in response.get('files', []):
            if parent_id is None:
                return result.get('id')
            else:
                if parent_id in result.get('parents'):
                    return result.get('id')
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return None


def upload_image(service, name, folder_id, source_filepath):
    file_metadata = {
        'name': name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(source_filepath, mimetype='image/jpeg', resumable=True)
    file = service\
        .files()\
        .create(body=file_metadata, media_body=media, fields='id')\
        .execute()
    return file.get('id')


def update_or_create_image_in_existing_dir(service, name, folder_id, source_filepath, logger):
    assert folder_id is not None
    img_id = get_file_id(service, name, folder_id)
    if img_id is None:
        upload_image(service, name, folder_id, source_filepath)
    else:
        logger.warning("file {} exists.".format(name))


def make_folder_if_not_exists(service, name, logger, parent_id=None):
    dir_id = get_folder_id(service, name, parent_id)
    if dir_id is None:
        logger.info("Creating folder {}".format(name))
        dir_id = make_folder(service, name, logger, parent_id=parent_id)
    else:
        logger.info("Folder {} exists".format(name))
    return dir_id


def make_folder(service, name, logger, parent_id=None):
    logger.info("Will create folder {} under {}".format(name, parent_id))
    if parent_id is None:
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
        }
    else:
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
    file = service\
        .files()\
        .create(body=file_metadata, fields='id')\
        .execute()
    return file.get('id')
