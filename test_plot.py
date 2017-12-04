#coding :utf-8
import numpy as np
# from imp import reload
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Db_conn import *
import datetime,os

db_conn = Conn_db()

def conn_machine_data():
    # file_dir = "./fifth_mid"
    # m_list = []
    # with open(file_dir) as flg:
    #     lines = flg.readlines()
    #     for l in lines:
    #         lines = l.strip('\n').strip('\t').split('\t')
    #         for i in lines:
    #             m_list.append(i)
    # print(len(m_list))
    # return m_list

    sql = "SELECT DLO_NO FROM scannerlist WHERE dlo_no='BL160138'"
    result = db_conn.run(sql)
    mid_list = []
    if result:
        for x in result:
            mid_list.append(x[0])
    print(len(mid_list))
    return mid_list

def group_every_mid_area(mid_list):
    area_list = ['HD','XN','HZ','HN','HB','LY','XB','DB']
    area_mid_dict = dict([(k,[]) for k in area_list])
    sql = "SELECT area  FROM scannerlist WHERE dlo_no='%s'"
    for a in area_mid_dict.keys():
        for m in mid_list:
            sql_str = sql%(m)
            result = db_conn.run(sql_str)
            if len(result)==0:
                area='HN'
            else:
                area = result[0][0]
            if area==a:
                area_mid_dict[a].append(m)
    return area_mid_dict

def plot_every_mid_data(group_mid_area_dict):
    sql = "SELECT scan_date,foot_length_original_left from dlo_3dscan where scan_id LIKE '%s' ORDER BY scan_date"
    file_str = 'C:\\Users\Administrator\Desktop\\fifth_\\%s'
    for g in group_mid_area_dict.keys():
        file_ = file_str%(g)
        # if os.path.exists(file_str)==False:
        #     os.makedirs(file_)
        if len(group_mid_area_dict[g]):
            for m in group_mid_area_dict[g]:
                sql_str = sql%(str(m)+'%')
                result = db_conn.run(sql_str)
                if len(result)>0:
                    scan_date= [y[0] for  y in result ]
                    print(m,scan_date)
                    scan_hour = calculate_time_space(scan_date)
                    scan_foot_length_original_left = [x[1] for x in result]
                    plt.axis([0,max(scan_hour)+10,0,max(scan_foot_length_original_left)+10])
                    plt.plot([2150, 2150], [0, max(scan_foot_length_original_left)+10], 'r--')
                    plt.scatter(scan_hour,scan_foot_length_original_left)
                    plt.grid(axis='y',linestyle = "-.",color = "r", linewidth = "0.5")
                    plt.xlabel('Hours')
                    plt.ylabel('foot_length_original_left')
                    plt.show()
                    # plt.savefig(file_+'\\'+str(m)+'.png')
                    # plt.close('all')


def calculate_time_space(date_list):
    origin_date = datetime.datetime(2017, 4, 1, 0, 0, 0)
    date_hour = []
    for d in date_list:
        timedelt = datetime.timedelta.total_seconds(d-origin_date)/(60*60)
        date_hour.append(timedelt)
    return date_hour

def test(mid_list):
    mid_str = '|'.join([str(x)+'%' for x in mid_list] )
    sql = "select scan_date from (select  SUBSTRING(scan_id,0,9) as sid,scan_date,row_number() over(partition by sid) as row  from dlo_3dscan  where  scan_id similar to '%s' ORDER by scan_date)a WHERE row <=1;"
    sql_str = sql%(mid_str)
    result = db_conn.run(sql_str)
    print(result)



if __name__ =="__main__":
    mid_list = conn_machine_data()
    group_mid_area_dict = group_every_mid_area(mid_list)
    plot_every_mid_data(group_mid_area_dict)
