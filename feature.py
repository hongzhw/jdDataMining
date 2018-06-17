import pandas as pd
import numpy as np

# 读取用户风险标签
uid_train = pd.read_csv('JDATA_TRAIN/uid_train.txt', sep='\t', header=None)
uid_train.columns = ['uid', 'label']

# 读取用户通话记录数据
voice_train = pd.read_csv('JDATA_TRAIN/voice_train.txt', sep='\t', header=None)
voice_train.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']

# 读取用户短信记录数据
sms_train = pd.read_csv('JDATA_TRAIN/sms_train.txt', sep='\t', header=None)
sms_train.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']

# 读取用户网站/App访问记录数据
wa_train = pd.read_csv('JDATA_TRAIN/wa_train.txt', sep='\t', header=None)
wa_train.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'data']

# 读取测试集的用户通话记录、用户短信记录、用户网站/App访问记录
voice_test = pd.read_csv('JDATA_TEST_B/voice_test_b.txt', sep='\t',header=None, names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('JDATA_TEST_B/sms_test_b.txt', sep='\t',header=None, names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('JDATA_TEST_B/wa_test_b.txt', sep='\t',header=None, names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

# 给测试集建立uid
ids = np.arange(7000,9000)
uid_test = pd.DataFrame()
i = []
for item in ids:
    i.append('u' + str(item))
uid_test['uid'] = i
uid_test.to_csv('JDATA_TEST_B/uid_test_b.txt', index=None)

voice = pd.concat([voice_train, voice_test], axis=0)
sms = pd.concat([sms_train, sms_test], axis=0)
wa = pd.concat([wa_train, wa_test], axis=0)


# 特征工程构建

# 通话记录个数和通话记录对端号码个数
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({
    'voice_total_count': 'count', 'voice_unique_count': lambda x: len(pd.unique(x))}).reset_index()

# 用户对端通话号码类型次数统计
voice_call_type = voice.groupby(
    ['uid', 'call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

# 用户发起通话和被叫通话的次数统计
voice_in_out = voice.groupby(
    ['uid', 'in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

# 用户对端通话号码长度的统计
voice_opp_len = voice.groupby(
    ['uid', 'opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)


# 用户对端通话号码的时长统计
# 时间转换函数，将字符串时间格式转换到以秒为单位的数字格式
def time_format(x):
    # 时间格式为 hh:mm:ss
    if str(x).__len__() == 6:
        return int(str(x)[0:2]) * 3600 + int(str(x)[2:4]) * 60 + int(str(x)[4:6])
    # 时间格式为 d:hh:mm:ss
    elif str(x).__len__() == 7:
        return int(str(x)[0]) * 86400 + int(str(x)[1:3]) * 3600 + int(str(x)[3:5]) * 60 + int(str(x)[5:7])
    # 时间格式为 dd:hh:mm:ss
    else:
        return int(str(x)[0:2]) * 86400 + int(str(x)[2:4]) * 3600 + int(str(x)[4:6]) * 60 + int(str(x)[6:8])

voice['call_start'] = voice.start_time.apply(time_format)
voice['call_end'] = voice.end_time.apply(time_format)

# 通话时长
voice['call_dura'] = voice['call_end'] - voice['call_start']
# 通话时长总和
voice_time_dura = voice.groupby(['uid'])['call_dura'].agg({'total_time': 'sum'}).reset_index()
# 通话时长各种数学统计值
voice_time = voice.groupby(['uid'])['call_dura'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('call_dura_').reset_index()

# 每日通话次数统计
def day(x):
    if str(x).__len__() == 6:
        return "00"
    elif str(x).__len__() == 7:
        return "0" + str(x)[0]
    else:
        return str(x)[0:2]

voice['voice_per_day'] = voice.start_time.apply(day)
voice_per_day = voice.groupby(['uid', 'voice_per_day'])['uid'].count().unstack().add_prefix('voice_day_').reset_index().fillna(0)

# 通话时长分时间段统计
# 1->通话时间少于半小时， 2->小于1小时, 3->小于2小时， 4->大于2小时
def dura_count_type(x):
    if x < 60 * 30:
        return 1
    elif x < 60 * 60:
        return 2
    elif x < 60 * 120:
        return 3
    else:
        return 4
voice['voice_dura_type'] = voice.call_dura.apply(dura_count_type)
voice_dura_type = voice.groupby(
    ['uid', 'voice_dura_type'])['uid'].count().unstack().add_prefix('voice_dura_type_').reset_index().fillna(0)


voice_feature = pd.merge(voice_opp_num, voice_call_type, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_in_out, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_opp_len, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_time_dura, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_time, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_per_day, how='left', on='uid')
voice_feature = pd.merge(voice_feature, voice_dura_type, how='left', on='uid')


# 短信个数总量和短信对端号码个数统计
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg(
    {'sms_total_count': 'count', 'sms_unique_count': lambda x: len(pd.unique(x))}).reset_index()

# 用户短信号码长度
sms_opp_len = sms.groupby(['uid', 'opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

# 接收短信和发送短信次数统计
sms_in_out = sms.groupby(['uid', 'in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)

# 用户不同opp_head记录数
sms_opp_head = sms.groupby(['uid'])['opp_head'].agg(
    {'sms_total_head_count': 'count', 'sms_unique_head_count': lambda x: len(pd.unique(x))}).reset_index()

# 每天的短信个数
sms['sms_per_day'] = sms.start_time.apply(day)
sms_per_day = sms.groupby(['uid', 'sms_per_day'])['uid'].count().unstack().add_prefix('sms_day_').reset_index().fillna(0)

sms_feature = pd.merge(sms_opp_head, sms_opp_num, how='left', on='uid')
sms_feature = pd.merge(sms_feature, sms_opp_len, how='left', on='uid')
sms_feature = pd.merge(sms_feature, sms_in_out, how='left', on='uid')
sms_feature = pd.merge(sms_feature, sms_per_day, how='left', on='uid')

# 访问网站/App总次数和访问的网站/App个数
wa_name = wa.groupby(['uid'])['wa_name'].agg(
    {'wa_name_count': 'count', 'wa_name_unique_count': lambda x: len(pd.unique(x))}).reset_index()

# 访问次数的各类数学统计
# 标准差、最大值、最小值、中位数、平均数、和
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_visit_cnt_').reset_index()

# 访问时长的各类数学统计
# 标准差、最大值、最小值、中位数、平均数、和
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_visit_dura_').reset_index()

# 上行流量的各类数学统计
# 标准差、最大值、最小值、中位数、平均数、和
up_flow = wa.groupby(['uid'])['up_flow'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_up_flow_').reset_index()

# 下行流量的各类数学统计
# 标准差、最大值、最小值、中位数、平均数、和
down_flow = wa.groupby(['uid'])['down_flow'].agg(
    ['std', 'max', 'min', 'median', 'mean', 'sum']).add_prefix('wa_down_flow_').reset_index()

# 每天访问网站/App次数
wa['wa_per_day'] = wa.date.apply(day)
wa_per_day = wa.groupby(['uid', 'wa_per_day'])['uid'].count().unstack().add_prefix('wa_day_').reset_index().fillna(0)

wa_feature = pd.merge(wa_name, visit_cnt, how='left', on='uid')
wa_feature = pd.merge(wa_feature, visit_dura, how='left', on='uid')
wa_feature = pd.merge(wa_feature, up_flow, how='left', on='uid')
wa_feature = pd.merge(wa_feature, down_flow, how='left', on='uid')
wa_feature = pd.merge(wa_feature, wa_per_day, how='left', on='uid')

train_feature = uid_train
train_feature = pd.merge(train_feature, voice_feature, how='left', on='uid')
train_feature = pd.merge(train_feature, sms_feature, how='left', on='uid')
train_feature = pd.merge(train_feature, wa_feature, how='left', on='uid')

test_feature = uid_test
test_feature = pd.merge(test_feature, voice_feature, how='left', on='uid')
test_feature = pd.merge(test_feature, sms_feature, how='left', on='uid')
test_feature = pd.merge(test_feature, wa_feature, how='left', on='uid')

# print(train_feature.dtypes)

train_feature.to_csv('train_featureV1.csv',index=None)
test_feature.to_csv('test_featureV1.csv',index=None)
