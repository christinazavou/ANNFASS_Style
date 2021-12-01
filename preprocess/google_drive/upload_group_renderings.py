from __future__ import print_function

import json
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm
import argparse
import logging


LOGGER = logging.getLogger(name="upload")
SCOPES = ['https://www.googleapis.com/auth/drive']
STYLES_TXT = "../../resources/STYLES.txt"


def set_logger_file(log_file):
    global LOGGER
    file_handler = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    for hdlr in LOGGER.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr, logging.FileHandler):
            LOGGER.removeHandler(hdlr)
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(logging.INFO)


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
            credentials = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
    return credentials


def find_folders_with_substring(substring):
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
            LOGGER.info('Found file: %s (%s)' % (file.get('name'), file.get('id')))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

# deletes(puts in trash)
# def delete_by_id(f_id):
#     r = service \
#         .files()\
#         .delete(fileId=f_id)\
#         .execute()


def get_folder_id(name, parent_id=None):
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


def get_file_id(name, parent_id=None):
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


def upload_image(name, folder_id, source_filepath):
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


def update_or_create_image_in_existing_dir(name, folder_id, source_filepath):
    assert folder_id is not None
    img_id = get_file_id(name, folder_id)
    if img_id is None:
        upload_image(name, folder_id, source_filepath)
    else:
        LOGGER.warning("file {} exists.".format(name))


def make_folder_if_not_exists(name, parent_id=None):
    dir_id = get_folder_id(name, parent_id)
    if dir_id is None:
        LOGGER.info("Creating folder {}".format(name))
        dir_id = make_folder(name, parent_id=parent_id)
    else:
        LOGGER.info("Folder {} exists".format(name))
    return dir_id


def make_folder(name, parent_id=None):
    LOGGER.info("Will create folder {} under {}".format(name, parent_id))
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


def continue_with_warning(render_dir, groups_dir, groups_file):
    if not os.path.exists(render_dir):
        LOGGER.warning("Render dir {} doesnt exist".format(render_dir))
        return True
    if not os.path.exists(groups_dir):
        LOGGER.warning("Groups dir {} doesnt exist".format(groups_dir))
        return True
    if not os.path.exists(groups_file):
        LOGGER.warning("Groups file {} doesnt exist".format(groups_file))
        return True
    return False


def upload_data(root_id, group_renderings_path, groups_path, buildings_csv):

    buildings = parse_buildings_csv(buildings_csv)

    with open(STYLES_TXT, "r") as fin:
        lines = fin.readlines()
        styles = [line.strip() for line in lines]
        LOGGER.info("styles: {}".format(styles))

    for building in buildings:
        group_render_dir = os.path.join(group_renderings_path, building)
        groups_dir = os.path.join(groups_path, building)
        groups_file = os.path.join(groups_dir, "groups.json")
        if continue_with_warning(group_render_dir, groups_dir, groups_file):
            continue
        LOGGER.info("Uploading for {}".format(building))

        with open(groups_file, "r") as fin:
            groups = json.load(fin)
        for group in groups:
            group_first_element = groups[group][0]
            group_style = [x for x in styles if x.lower() in group_first_element.lower()][0]  # always one style

            for file in os.listdir(group_render_dir):
                if "group{}_".format(group) in file:
                    if ".jpg" in file and not 'grouped_' in file and not "discard" in file:
                        style_dir_id = make_folder_if_not_exists(group_style, root_id)
                        style_building_dir_id = make_folder_if_not_exists(building, style_dir_id)

                        filepath = os.path.join(group_render_dir, file)
                        update_or_create_image_in_existing_dir(file, style_building_dir_id, filepath)


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    LOGGER.info("buildings to process: {}".format(buildings))
    return buildings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--gdrepo", default="ANNFASS_Buildings_semifinal", type=str)
    parser.add_argument("--groups_renderings_path", default="groups_renderings", type=str)
    parser.add_argument("--groups_path", default="groups", type=str)
    parser.add_argument("--buildings_csv", required=True, type=str)
    parser.add_argument("--logs_dir", required=True, type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, "upload_group_renderings.log")
    set_logger_file(_log_file)
    LOGGER.info("root_dir: {}".format(args.root_dir))
    LOGGER.info("gdrepo: {}".format(args.gdrepo))
    LOGGER.info("groups_renderings_path: {}".format(args.groups_renderings_path))
    LOGGER.info("groups_path: {}".format(args.groups_path))
    LOGGER.info("buildings_csv: {}".format(args.buildings_csv))

    service = build('drive', 'v3', credentials=get_and_save_access())

    root_dir_id = make_folder_if_not_exists(args.gdrepo, parent_id=get_folder_id('ANNFASS'))

    upload_data(root_dir_id,
                os.path.join(args.root_dir, args.groups_renderings_path),
                os.path.join(args.root_dir, args.groups_path),
                args.buildings_csv)

