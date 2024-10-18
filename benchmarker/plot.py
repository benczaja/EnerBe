import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pdb

class Plotter():
    def __init__(self):
        self.data = {}
        self.plot_data = {}
 
    def load_data(self,result_csv):
        self.data = pd.read_csv(result_csv,low_memory=False)
        self.plot_data = self.data

    def TPE_plot(self, x, *args, **kwargs):
        style = kwargs.get('style', None)
        sort_by = kwargs.get('sort_by', None)

        for name in self.plot_data["NAME"].unique():
            for algo in self.plot_data["ALGO"].unique():
            
                plot_data = self.plot_data[(self.plot_data['NAME'] == name) & (self.plot_data['ALGO'] == algo)]

                if len(plot_data) == 0:
                    continue
                print("Plotting\nName: " + name + "\nAlgo: " + algo)

                title = name + "_" + algo


                if sort_by:
                    plot_data = plot_data.sort_values(by=sort_by)
        
                f, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 10))

                sns.lineplot(x=x, y="CPU_TIME",  hue="CPU_NAME", style=style,  data=plot_data, markers=True,ax=axs[0,0])
                sns.lineplot(x=x, y="CPU_WATTS", hue="CPU_NAME", style=style,  data=plot_data, markers=True,ax=axs[1,0],legend=False)
                sns.lineplot(x=x, y="CPU_JOULES",hue="CPU_NAME", style=style,  data=plot_data, markers=True,ax=axs[2,0],legend=False)

                sns.lineplot(x=x, y="GPU_TIME",  hue="GPU_NAME", style=style,  data=plot_data, markers=True,ax=axs[0,1])
                sns.lineplot(x=x, y="GPU_WATTS", hue="GPU_NAME", style=style,  data=plot_data, markers=True,ax=axs[1,1],legend=False)
                sns.lineplot(x=x, y="GPU_JOULES",hue="GPU_NAME", style=style,  data=plot_data, markers=True,ax=axs[2,1],legend=False)

                axs[0,1].set_ylabel("GPU Time (s)")
                axs[1,1].set_ylabel("GPU Power (W)")
                axs[2,1].set_ylabel("GPU Energy (J)")

                axs[0,0].set_ylabel("CPU Time (s)")
                axs[1,0].set_ylabel("CPU Power (W)")
                axs[2,0].set_ylabel("CPU Energy (J)")

                f.suptitle(title)
                axs[0,0].legend(loc=2, prop={'size': 6})
                axs[0,1].legend(loc=2, prop={'size': 6})

                plt.tight_layout()

                plt.savefig("TPE_" + name + "_" + algo +".png",dpi=200)

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
        

