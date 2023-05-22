from config.const import const_config

plate_chr = const_config.DATASET.PLATE_CHR

class PutFuction():
    def __init__(self,info=None,type = 'normal'):
        self.type = type
        self.info = info
        self.plateStr = plate_chr
        self.plateDict ={}
        for i in range(len(list(self.plateStr))):
            self.plateDict[self.plateStr[i]]=i
    
    def is_str_right(self,plate_name):
        for str_ in plate_name:
            if str_ not in self.plateStr:
                return False
        return True
    
    def putLabel(self, oriLabelstr = None):
        if " " in oriLabelstr:
            return None
        if not self.is_str_right(oriLabelstr):
            return None
        # normal:正常label
        # total: 整个label替换为poison，info里面poison为整体的后门
        # part: 部分label替换为poison，info里面index是一个dict，key是需要替换的位置，value为替换的值
        if self.type == 'normal':
            labelstr = ' '
            strList = list(oriLabelstr)
            for i in range(len(strList)):
                labelstr+=str(self.plateDict[strList[i]])+" "
        elif self.type == 'total':
            labelstr = ' '
            for i in range(len(self.info['poison'])):
                labelstr+=str(self.plateDict[self.info['poison'][i]])+" "
        elif self.type == 'part':
            labelstr = ' '
            strList = list(oriLabelstr)
            for i in range(len(strList)):
                if str(i) in self.info['index']:
                    labelstr+=str(self.plateDict[self.info['index'][str(i)]])+" "
                else:
                    labelstr+=str(self.plateDict[strList[i]])+" "
        else:
            print("未定义数据处理方式")
            exit(1)
        return labelstr