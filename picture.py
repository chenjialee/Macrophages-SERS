import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


class setfig():
    '''
       在绘图前对字体类型、字体大小、分辨率、线宽、输出格式进行设置.
       para column = 1. 半栏图片 7*6cm
                     2. 双栏长图 14*6cm
       x轴刻度默认为整数
    '''

    def __init__(self, column):
        self.column = column  # 设置栏数
        # 对尺寸和 dpi参数进行调整
        plt.rcParams['figure.dpi'] = 300

        # 字体调整
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.weight'] = 'light'  # 字体的粗细：normal,bold,light
        plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
        plt.rcParams['axes.titlesize'] = 9  # 标题字体大小
        plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
        plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
        plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
        plt.rcParams['legend.fontsize'] = 7  # 图例字体的大小

        # 线条调整
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['lines.linewidth'] = 1

        # 刻度在内，设置刻度字体大小
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # 设置输出格式为SVG（默认格式为SVG）
        plt.rcParams['savefig.format'] = 'svg'
        plt.rcParams['figure.autolayout'] = True

    def set_tickfont(self):
        tick_font = font_manager.FontProperties(family='Arial', size=7.0)
        ax1 = plt.gca()  # 获取当前图像的坐标轴
        for labelx in ax1.get_xticklabels():
            labelx.set_fontproperties(tick_font)
        for labely in ax1.get_yticklabels():
            labely.set_fontproperties(tick_font)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数

    def set_global_font(self):
        # 设置全局字体
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.weight'] = 'bold'

    def show(self):
        # 应用字体和刻度设置
        self.set_global_font()
        self.set_tickfont()

        # 创建图形对象
        fig, ax = plt.subplots()

        # 改变图像大小
        cm_to_inch = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
        if self.column == 1:
            fig.set_size_inches(7 * cm_to_inch, 6 * cm_to_inch)  # 半栏图形
        else:
            fig.set_size_inches(14 * cm_to_inch, 6 * cm_to_inch)  # 双栏长图

        # plt.subplots_adjust(top=0.95, bottom=0.2, left=0, right=1)  # 设置上边距和下边距，调整高度空间
        # plt.tight_layout()
        return fig, ax  # 返回图形对象 fig 和坐标轴对象 ax


