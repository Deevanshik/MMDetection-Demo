# Copyright (c) Open-MMLab. All rights reserved.

__version__ = '2.3.0rc0'
short_version = '2.3.0rc0'


def parse_version_info():
    version_info = []
    for x in short_version.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info()
