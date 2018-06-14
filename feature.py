import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 数据读取 读取用户风险标签 分隔符为制表符\t
uid_train = pd.read_csv('JDATA_TRAIN/uid_train.txt', sep='\t', header=None)
uid_train.columns = ['uid', 'label']


# 数据读取 读取用户的短信记录 分隔符为\t
sms_train = pd.read_csv('JDATA_TRAIN/sms_train.txt', sep='\t', header=None)
sms_train.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']


# 数据读取 读取用户的通话记录 分隔符\t
voice_train = pd.read_csv('JDATA_TRAIN/voice_train.txt', sep='\t', header=None)
voice_train.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']

# 数据读取 读取用户的上网记录 分割符\t
wa_train = pd.read_csv('JDATA_TRAIN/wa_train.txt', sep='\t', header=None)
wa_train.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'data']


# 读取测试文件
voice_test = pd.read_csv('JDATA_TEST_B/voice_test_b.txt', sep='\t',header=None, names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('JDATA_TEST_B/sms_test_b.txt', sep='\t',header=None, names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('JDATA_TEST_B/wa_test_b.txt', sep='\t',header=None, names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

# 数据整理
uid_test = pd.DataFrame({'uid': pd.unique(wa_test['uid'])})
uid_test.to_csv('JDATA_TEST_B/uid_test_b.txt', index=None)

voice = pd.concat([voice_train, voice_test], axis=0)
sms = pd.concat([sms_train, sms_test], axis=0)
wa = pd.concat([wa_train, wa_test], axis=0)

# 数据分析

# 建立关于用户对端通话号码的特征 即用户总通话数和用户的总对端号码数
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({
    'voice_total_count': 'count', 'voice_unique_count': lambda x: len(pd.unique(x))}).reset_index()

# 建立关于用户对端通话号码的前n位的特征 包括各类号码前n位的次数以及通话最多的前n位以及通话位数的统计
voice_opp_head = voice.groupby(['uid'])['opp_head'].agg(
    {'unique_count': lambda x: len(pd.unique(x)), 'max_count': lambda x: x.value_counts().index[0]}).add_prefix('voice_opp_head_').reset_index()

# 巨坑注意object需要进行转换以及相关的对象


def opp_head_size(x):
    return str(x).__len__()


voice_opp_head_x = voice.opp_head.apply(opp_head_size)
voice['opp_head_size'] = voice_opp_head_x
voice_opp_head_y = voice.groupby(['uid', 'opp_head_size'])['uid'].count().unstack().add_prefix('voice_opp_head_').reset_index().fillna(0)
voice_opp_head = pd.merge(voice_opp_head, voice_opp_head_y, how='left', on='uid')

# 建立用户对端通话号码的时长统计 包括用户对端通话号码的总时长以及平均时长


def time_count(x):
    if str(x).__len__() == 6:
        return int(str(x)[0:2]) * 3600 + int(str(x)[2:4]) * 60 + int(str(x)[4:6])
    elif str(x).__len__() == 7:
        return int(str(x)[0]) * 86400 + int(str(x)[1:3]) * 3600 + int(str(x)[3:5]) * 60 + int(str(x)[5:7])
    else:
        return int(str(x)[0:2]) * 86400 + int(str(x)[2:4]) * 3600 + int(str(x)[4:6]) * 60 + int(str(x)[6:8])


voice_x = voice.start_time.apply(time_count)
voice_y = voice.end_time.apply(time_count)
voice['start_count'] = voice_x
voice['end_count'] = voice_y

voice['time_long'] = voice['end_count'] - voice['start_count']

voice_time_long = voice.groupby(['uid'])['time_long'].agg({'total_time': 'sum', 'average_time': lambda x: x.sum() / x.count()}).reset_index()


# 建立用户对端通话号码长度的统计
voice_opp_len = voice.groupby(
    ['uid', 'opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

# 建立用户对端通话号码类型的统计
voice_call_type = voice.groupby(
    ['uid', 'call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

# 用户通话主叫和被叫的次数统计
# 这里要用到多重索引构建的信息, 通过将uid和in_out两重索引, 建立出uid和0 1 之间的关系, 由于多重索引构建出来的是stack状态
# 因此要进行unstack处理, 之后用0填充缺失的部分
voice_in_out = voice.groupby(
    ['uid', 'in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice_feature = pd.merge(voice_opp_num, voice_opp_head, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_time_long, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_opp_len, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_call_type, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_in_out, how='left', on='uid')


# 建立用户短信对端号码的统计 总数和号码数
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg(
    {'sms_total_count': 'count', 'sms_unique_count': lambda x: len(pd.unique(x))}).reset_index()

# 建立用户对端短信号码的前n位的特征 包括各类号码前n位的次数以及短信最多的前n位以及短信位数的统计
sms_opp_head = sms.groupby(['uid'])['opp_head'].agg(
    {'unique_count': lambda x: len(pd.unique(x)), 'max_count': lambda x: x.value_counts().index[0]}).add_prefix('sms_opp_head_').reset_index()

sms_opp_head_x = sms.opp_head.apply(opp_head_size)
sms['opp_head_size'] = sms_opp_head_x
sms_opp_head_y = sms.groupby(['uid', 'opp_head_size'])['uid'].count().unstack().add_prefix('voice_opp_head_').reset_index().fillna(0)
sms_opp_head = pd.merge(sms_opp_head, sms_opp_head_y, how='left', on='uid')

# 用户短信号码长度统计
sms_opp_len = sms.groupby(['uid', 'opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

# 用户短信号码主被叫次数
sms_in_out = sms.groupby(['uid', 'in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)

sms_feature = pd.merge(sms_opp_head, sms_opp_num, how='left', on='uid')
sms_feature = pd.merge(sms_feature, sms_opp_len, how='left', on='uid')
sms_feature = pd.merge(sms_feature, sms_in_out, how='left', on='uid')

# 网站名称
wa_name = wa.groupby(['uid'])['wa_name'].agg(
    {'wa_name_count': 'count', 'wa_name_unique_count': lambda x: len(pd.unique(x))}).reset_index()

# 访问次数
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_visit_cnt_').reset_index()
# print(visit_cnt)

# 访问时长
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_visit_dura_').reset_index()

# 上行流量统计
up_flow = wa.groupby(['uid'])['up_flow'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_up_flow_').reset_index()

# 下行流量统计
down_flow = wa.groupby(['uid'])['down_flow'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_down_flow_').reset_index()

wa_feature = pd.merge(wa_name, visit_cnt, how='left', on='uid')
wa_feature = pd.merge(wa_feature, visit_dura, how='left', on='uid')
wa_feature = pd.merge(wa_feature, up_flow, how='left', on='uid')
wa_feature = pd.merge(wa_feature, down_flow, how='left', on='uid')

train_feature = uid_train
train_feature = pd.merge(train_feature, voice_feature, how='left', on='uid')
train_feature = pd.merge(train_feature, sms_feature, how='left', on='uid')
train_feature = pd.merge(train_feature, wa_feature, how='left', on='uid')

test_feature = uid_test
test_feature = pd.merge(test_feature, voice_feature, how='left', on='uid')
test_feature = pd.merge(test_feature, sms_feature, how='left', on='uid')
test_feature = pd.merge(test_feature, wa_feature, how='left', on='uid')

print(train_feature.dtypes)
print(train_feature.dtypes)


train_feature.to_csv('train_featureV1.csv',index=None)
test_feature.to_csv('test_featureV1.csv',index=None)

