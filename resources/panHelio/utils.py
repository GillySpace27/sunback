"""Utility methods for SPRINTS donwload scripts
"""
import os
import re
import json
import shutil
import pickle
import logging
import requests
import ftputil
from pathlib import Path
from bs4 import BeautifulSoup


def download_http(base_url, target_dir, history_file, file_pattern=None, overwrite=False):
    """Download files recursively via HTTP or HTTPS protocol
    The list of downloaded files will be appended to the 'history_file'
    Parameters
    ----------
    base_url: str
        the URL to download files from
    target_dir: str
        the local directory to download the files to
    history_file: str
        the pickle file name containing list of already downloaded files (so that we don't download them again)
    file_pattern: str (optional)
        the regular expression to match file names.  Default will match all files.
    overwrite: boolean (Default is False)
        whether to ownload file that has already been downloaded
    """
    # create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # list of file names to exclude
    exclude_list = read_history(history_file)

    # flag to indicate whether any file is downloaded
    downloaded = False

    try:
        # parse the URL page and retrieve links
        logging.info('Inspecting %s', base_url)

        page = requests.get(base_url).content
        bs_obj = BeautifulSoup(page, 'html.parser')
        links = bs_obj.find_all('a', href=True)

        for link in links:
            url = link['href']

            # directory
            if is_dir(url):
                if url not in base_url:
                    dir_url = base_url + url
                    # recursively download the directory
                    download_http(dir_url, target_dir,
                                  history_file, file_pattern)
                    # update the exclude_list
                    exclude_list = read_history(history_file)

            else:
                # file - check if file name matches
                if not file_pattern or re.search(file_pattern, url):
                    file_name = os.path.basename(url)
                    file_url = base_url + url
                    target_file = os.path.join(target_dir, file_name)

                    # download the file if it's not excluded nor already downloaded
                    if overwrite or (file_name not in exclude_list and not os.path.exists(target_file)):
                        logging.info('Downloading %s', file_url)

                        try:
                            with requests.get(file_url, stream=True) as r:
                                if (target_file.endswith('.json')):
                                    with open(target_file, 'w') as f:        
                                        json.dump(json.loads(r.text), f)
                                else:
                                    with open(target_file, 'wb') as f:        
                                        shutil.copyfileobj(r.raw, f)

                            # add file to exclude list
                            if file_name not in exclude_list:
                                exclude_list.append(file_name)
                                downloaded = True

                        except Exception as ex:
                            logging.error('Download failed: %s', str(ex))

    except Exception as e:
        logging.error('Inspect URL failed: ' + str(e))

    # save download history only if new files are downloaded
    if downloaded:
        save_history(exclude_list, history_file)

def download_http_direct(base_url, target_dir, history_file, file_name, overwrite=False):
    """Download specific files via HTTP or HTTPS protocol
    The list of downloaded files will be appended to the 'history_file'
    Parameters
    ----------
    base_url: str
        the URL to download files from
    target_dir: str
        the local directory to download the files to
    history_file: str
        the pickle file name containing list of already downloaded files (so that we don't download them again)
    file_name: str
        the name of the specific file to download
    overwrite: boolean (Default is False)
        whether to ownload file that has already been downloaded
    """
    # create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # list of file names to exclude
    exclude_list = read_history(history_file)

    # flag to indicate whether any file is downloaded
    downloaded = False

    try:
        logging.info('Inspecting %s', base_url)
        file_url = base_url+file_name
        target_file = target_dir+'/'+file_name
        # check to see if the file has been downloaded
        # if not, download it 
        if overwrite or (file_name not in exclude_list and not os.path.exists(target_file)):
            logging.info('Downloading %s', file_url)
            try:
                print(file_url)
                with requests.get(file_url, stream=True) as r:
                    if (target_file.endswith('.json')):
                        with open(target_file, 'w') as f:        
                            json.dump(json.loads(r.text), f)
                    else:
                        with open(target_file, 'wb') as f:        
                            shutil.copyfileobj(r.raw, f)

                # add file to exclude list
                if target_file not in exclude_list:
                    exclude_list.append(target_file)
                    downloaded = True

            except Exception as ex:
                logging.error('Download failed: %s', str(ex))

    except Exception as e:
        logging.error('Inspect URL failed: ' + str(e))

    # save download history only if new files are downloaded
    if downloaded:
        save_history(exclude_list, history_file)

def download_ftp(ftp_host, target_dir, history_file, file_pattern=None, ftp_dir='/', ftp_user='anonymous', ftp_password='anonymous', overwrite=False):
    """Download files recursively from an FTP or FTPS server
    The list of downloaded files will be appended to the 'history_file'
    Parameters
    ----------
    ftp_host: str
        The FTP host name or IP address
    ftp_dir: str (optional)
        The FTP root directory to fetch files from
    ftp_user: str (optional)
        The FTP user name if any
    ftp_password: str (optional)
        The FTP password if any
    target_dir: str
        the local directory to download the files to
    history_file: str
        the pickle file name containing list of already downloaded files (so that we don't download them again)
    file_pattern: str (optional)
        the regular expression to match file names.  Default will match all files.
    overwrite: boolean (Default is False)
        whether to ownload file that has already been downloaded.  Set to True for near real time data.
    """
    # create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # list of file names to exclude
    exclude_list = read_history(history_file)

    # flag to indicate whether any file is downloaded
    downloaded = False

    try:
        # walking the FTP directory
        logging.info('Connecting to FTP host %s', ftp_host)

        with ftputil.FTPHost(ftp_host, ftp_user, ftp_password) as ftp:
            logging.info('Fetching files recusively from directory %s', ftp_dir)
            recursive = ftp.walk(ftp_dir, topdown=True, onerror=None)

            for root, dirs, files in recursive:
                for file_name in files:
                    # check if file_name matches
                    if not file_pattern or re.search(file_pattern, file_name):
                        src_file = ftp.path.join(root, file_name)
                        target_file = os.path.join(target_dir, file_name)

                        # download the file if 'overwrite' option is on OR it's not excluded nor already downloaded
                        if overwrite or (file_name not in exclude_list and not os.path.exists(target_file)):
                            logging.info('Downloading %s', src_file)
                            ftp.download(src_file, target_file)

                            if file_name not in exclude_list:
                                exclude_list.append(file_name)
                                downloaded = True

    except Exception as e:
        logging.error('Fetching files from %s failed: %s', ftp_host, str(e))

    # save download history only if new files are downloaded
    if downloaded:
        save_history(exclude_list, history_file)


def is_dir(url):
    """Check if an URL is a directory
    Parameters
    ----------
    url: str
        the URL to check
    Return
    ------
    boolean
        True if URL is a directory; false otherwise
    """
    if(url.endswith('/')):
        return True
    else:
        return False


def read_history(history_file):
    """Parse the download history pickle file into a list of strings
    Parameters
    ----------
    history_file: str
        the pickle file containing the download history
    Return
    ------
    list
        list of strings representing the file names
    """
    exclude_list = []
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            exclude_list = pickle.load(f)

    return exclude_list


def save_history(file_list, history_file):
    """Save the download history to a pickle file
    Parameters
    ----------
    file_list: list     
        list of file names
    history_file: str
        the pickle file to save the download history to
    """
    if not os.path.exists(os.path.dirname(history_file)):
        Path(os.path.dirname(history_file)).mkdir(parents=True, exist_ok=True)

    logging.info('Saving download history to %s', history_file)
    with open(history_file, 'wb') as f:
        pickle.dump(file_list, f)
