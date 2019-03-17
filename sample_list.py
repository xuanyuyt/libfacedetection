# -*- coding=utf-8 -*-
import os

if __name__ == '__main__':
    # ================================================================== #
    #                         Select Sample List                         #
    # ================================================================== #
    output = open('ALL.txt', 'w')
    with open('Path_Xmls.txt', 'r') as fp:
        for oneFile in fp:
            xmlname = oneFile.strip()
            jpgname = xmlname.replace('.xml', '.jpg').replace('Annotations', 'JPEGImages')
            output.write(jpgname)
            output.write('\t')
            output.write(xmlname)
            output.write('\n')
    output.close()
