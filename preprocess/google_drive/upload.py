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


def continue_with_warning(viewpoints_dir, mat_render_dir, fre_render_dir, groups_dir, groups_file):
    if not os.path.exists(viewpoints_dir):
        LOGGER.warning("Viewpoints dir {} doesnt exist".format(viewpoints_dir))
        return True
    if not os.path.exists(mat_render_dir) and not os.path.exists(fre_render_dir):
        LOGGER.warning("Renders dirs {},{} dont exist".format(mat_render_dir, fre_render_dir))
        return True
    if not os.path.exists(groups_dir):
        LOGGER.warning("Groups dir {} doesnt exist".format(groups_dir))
        return True
    if not os.path.exists(groups_file):
        LOGGER.warning("Groups file {} doesnt exist".format(groups_file))
        return True
    return False


def upload_data(root_id, renders_path, groups_path, viewpoints_path, buildings_csv):

    buildings = parse_buildings_csv(buildings_csv)

    with open(STYLES_TXT, "r") as fin:
        lines = fin.readlines()
        styles = [line.strip() for line in lines]
        LOGGER.info("styles: {}".format(styles))

    dir_ids = {}
    groups_dir_id = make_folder_if_not_exists("Groups", root_id)

    for building in buildings:
        viewpoints_dir = os.path.join(viewpoints_path, building)
        mat_render_dir = os.path.join(renders_path, "materials_on_daylight", building)
        fre_render_dir = os.path.join(renders_path, "freestyle", building)
        groups_dir = os.path.join(groups_path, building)
        groups_file = os.path.join(groups_dir, "groups.json")
        if continue_with_warning(viewpoints_dir, mat_render_dir, fre_render_dir, groups_dir, groups_file):
            continue
        LOGGER.info("Uploading for {}".format(building))
        for f in os.listdir(groups_dir):
            if "_grouped_" in f:
                use_name = "{}_{}".format(building, f)
                filepath = os.path.join(groups_dir, f)
                update_or_create_image_in_existing_dir(use_name, groups_dir_id, filepath)

        with open(groups_file, "r") as fin:
            groups = json.load(fin).keys()
        for group in groups:
            viewpoints_group_dir = os.path.join(viewpoints_dir, "group_{}".format(group))
            mat_render_group_dir = os.path.join(mat_render_dir, "group_{}".format(group))
            fre_render_group_dir = os.path.join(fre_render_dir, "group_{}".format(group))
            if not os.path.exists(viewpoints_group_dir):
                continue
            if not os.path.exists(mat_render_group_dir) and not os.path.exists(fre_render_group_dir):
                continue
            for file in os.listdir(viewpoints_group_dir):
                assert ".jpg" in file
                try:
                    current_style = [x for x in styles if x.lower() in file.lower()][0]  # always one element
                    LOGGER.info("current_style = {}, file = {}".format(current_style, file))
                except:
                    LOGGER.warning("No style found for file {} in group {} and building {}".format(file, group, building))
                    continue

                if current_style not in dir_ids:
                    style_dir_id = make_folder_if_not_exists(current_style, root_id)
                    style_discard_dir_id = make_folder_if_not_exists("Discarded", style_dir_id)
                    style_select_dir_id = make_folder_if_not_exists("Selected", style_dir_id)
                    dir_ids[current_style] = (style_dir_id, style_discard_dir_id, style_select_dir_id)

                if "discard" in file:
                    parent_id = dir_ids[current_style][1]
                    filepath = os.path.join(viewpoints_group_dir, file)
                    use_name = "{}_g{}_{}.jpg".format(building, group, file.replace(".jpg", ""))
                    update_or_create_image_in_existing_dir(use_name, parent_id, filepath)
                else:
                    parent_id = dir_ids[current_style][2]
                    filepath = os.path.join(mat_render_group_dir, file)
                    if os.path.exists(filepath):
                        use_name = "{}_g{}_{}_{}.jpg".format(building, group, file.replace(".jpg", ""), "mat")
                        update_or_create_image_in_existing_dir(use_name, parent_id, filepath)
                    filepath = os.path.join(fre_render_group_dir, file)
                    if os.path.exists(filepath):
                        use_name = "{}_g{}_{}_{}.jpg".format(building, group, file.replace(".jpg", ""), "fre")
                        update_or_create_image_in_existing_dir(use_name, parent_id, filepath)


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
    parser.add_argument("--gdrepo", default="BUILDNET_Buildings_tmp", type=str)
    parser.add_argument("--render_path", default="renderings", type=str)
    parser.add_argument("--viewpoints_path", default="viewpoints", type=str)
    parser.add_argument("--groups_path", default="groups", type=str)
    parser.add_argument("--buildings_csv", required=True, type=str)
    parser.add_argument("--logs_dir", default="upload.log", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    set_logger_file(_log_file)
    LOGGER.info("root_dir: {}".format(args.root_dir))
    LOGGER.info("gdrepo: {}".format(args.gdrepo))
    LOGGER.info("render_path: {}".format(args.render_path))
    LOGGER.info("groups_path: {}".format(args.groups_path))
    LOGGER.info("buildings_csv: {}".format(args.buildings_csv))

    service = build('drive', 'v3', credentials=get_and_save_access())

    root_dir_id = make_folder_if_not_exists(args.gdrepo, parent_id=get_folder_id('ANNFASS'))

    upload_data(root_dir_id,
                os.path.join(args.root_dir, args.render_path),
                os.path.join(args.root_dir, args.groups_path),
                os.path.join(args.root_dir, args.viewpoints_path),
                args.buildings_csv)

