import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class Plotter():
    def __init__(self):
        self.data = {}
        self.plot_data = {}
 
    def load_data(self,result_csv):
        self.data = pd.read_csv(result_csv,low_memory=False)
        self.plot_data = self.data

    def CPU_TPE_plot(self,x,*args, **kwargs):
        hue = kwargs.get('hue', None)
        style = kwargs.get('style', None)
        sort_by = kwargs.get('sort_by', None)
        title = kwargs.get('title', None)

        plot_data = self.plot_data
        print("Plotting Name: " + plot_data["NAME"].unique())

        if sort_by:
            plot_data = plot_data.sort_values(by=sort_by)
        
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

        sns.lineplot(x=x, y="CPU_time",  hue=hue, style=style,  data=plot_data, ax=ax1)
        sns.lineplot(x=x, y="CPU_power", hue=hue, style=style,  data=plot_data, ax=ax2,legend=False)
        sns.lineplot(x=x, y="CPU_energy",hue=hue, style=style,  data=plot_data, ax=ax3,legend=False)

        ax1.set_ylabel("CPU Time (s)")
        ax2.set_ylabel("CPU Power (W)")
        ax3.set_ylabel("CPU Energy (J)")

        ax1.set_title(title)

        plt.tight_layout()

        plt.savefig("CPU_TPE.png",dpi=200)

    def GPU_TPE_plot(self,x,*args, **kwargs):
        hue = kwargs.get('hue', None)
        style = kwargs.get('style', None)
        sort_by = kwargs.get('sort_by', None)
        title = kwargs.get('title', None)

        plot_data = self.plot_data
        plot_data["PRECISION"] = plot_data["PRECISION"]*8 

        if sort_by:
            plot_data = plot_data.sort_values(by=sort_by)
        
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

        sns.lineplot(x=x, y="GPU_TIME",  hue=hue, style=style,  data=plot_data, ax=ax1)
        sns.lineplot(x=x, y="GPU_WATTS", hue=hue, style=style,  data=plot_data, ax=ax2,legend=False)
        sns.lineplot(x=x, y="GPU_JOULES",hue=hue, style=style,  data=plot_data, ax=ax3,legend=False)

        ax1.set_ylabel("GPU Time (s)")
        ax2.set_ylabel("GPU Power (W)")
        ax3.set_ylabel("GPU Energy (J)")

        ax1.set_title(title)

        plt.tight_layout()

        plt.savefig("GPU_TPE.png",dpi=200)
        

