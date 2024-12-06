import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pdb

class Plotter():
    def __init__(self):
        self.data = {}
        self.plot_data = {}

        self.arch_palette ={
            # GPUs (with hosts)
            "NVIDIA A100-SXM4-40GB": "tab:blue",
            "Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz (x2)": "tab:blue", # host for A100

            "NVIDIA H100": "tab:green",
            "AMD EPYC 9334 32-Core Processor (x2)": "tab:green", # host for H100s

            'NVIDIA TITAN RTX': 'tab:orange',
            'Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz (x2)': 'tab:orange', # host for TITAN RTX
            'Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz (x2)': 'tab:orange',

            'NVIDIA GH200 480GB': "tab:purple",
            'NVIDIA GH200 480GB (x1)': "tab:purple",

            "Instinct MI210": "tab:orange",

            "AMD Instinct MI300A Accelerator (x4)": "tab:red",
            "AMD Instinct MI300A Accelerator": "tab:red",
            
            #CPUs
            "AMD EPYC 7H12 64-Core Processor (x2)": "tab:pink",
            "AMD EPYC 7H12 64-Core Processor (x1)": "tab:pink",

            "AMD EPYC 9654 96-Core Processor (x1)": "tab:cyan",
            "AMD EPYC 9654 96-Core Processor (x2)": "tab:cyan",

            "Graviton3 (x1)": "tab:grey"
            }
 
    def load_data(self,result_csv):
        self.data = pd.read_csv(result_csv,low_memory=False)
        self.plot_data = self.data
        self.plot_data["PRECISION"] = self.plot_data["PRECISION"]*8  

    def TPE_plot(self, x, *args, **kwargs):
        style = kwargs.get('style', None)
        sort_by = kwargs.get('sort_by', None)

        self.plot_data['CPU_NAME'] = self.plot_data['CPU_NAME'] + " (x" +self.plot_data['Sockets'].astype(str) + ")"
        for name in self.plot_data["NAME"].unique():
            for algo in self.plot_data["ALGO"].unique():
            
                plot_data = self.plot_data[(self.plot_data['NAME'] == name) & (self.plot_data['ALGO'] == algo)]

                if len(plot_data) == 0:
                    continue
                print("Plotting\nName: " + name + "\nAlgo: " + algo)

                title = name + "_" + algo

                if "Palabos" in name:
                    style="Sockets"
                else:
                    style = kwargs.get('style', None)

                if sort_by:
                    plot_data = plot_data.sort_values(by=sort_by)

                f, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 10))

                sns.lineplot(x=x, y="CPU_TIME",  hue="CPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[0,0])
                sns.lineplot(x=x, y="GPU_TIME",  hue="GPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[0,1])
                
                sns.lineplot(x=x, y="CPU_WATTS", hue="CPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[1,0],legend=False)
                sns.lineplot(x=x, y="GPU_WATTS", hue="GPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[1,1],legend=False)

                sns.lineplot(x=x, y="CPU_JOULES",hue="CPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[2,0],legend=False)
                sns.lineplot(x=x, y="GPU_JOULES",hue="GPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[2,1],legend=False)

                axs[0,1].set_ylabel("GPU TIME (s)")
                axs[1,1].set_ylabel("GPU Power (W)")
                axs[2,1].set_ylabel("GPU Energy (J)")

                axs[0,0].set_ylabel("CPU TIME (s)")
                axs[1,0].set_ylabel("CPU Power (W)")
                axs[2,0].set_ylabel("GFLOP/J")

                f.suptitle(title)
                axs[0,0].legend(loc=2, prop={'size': 6}, bbox_to_anchor=[0, 1.6])
                axs[0,1].legend(loc=2, prop={'size': 6}, bbox_to_anchor=[0, 1.6])

                plt.tight_layout()

                plt.savefig("TPE_" + name + "_" + algo +".png",dpi=200)

    def PPE_plot(self, x, *args, **kwargs):
        style = kwargs.get('style', None)
        sort_by = kwargs.get('sort_by', None)

        #self.plot_data['CPU_NAME'] = self.plot_data['CPU_NAME'] + " (x" +self.plot_data['Sockets'].astype(str) + ")"
        for name in self.plot_data["NAME"].unique():
            for algo in self.plot_data["ALGO"].unique():
            
                plot_data = self.plot_data[(self.plot_data['NAME'] == name) & (self.plot_data['ALGO'] == algo)]

                if len(plot_data) == 0:
                    continue
                print("Plotting\nName: " + name + "\nAlgo: " + algo)

                title = name + "_" + algo

                if "Palabos" in name:
                    style="Sockets"
                else:
                    style = kwargs.get('style', None)

                if sort_by:
                    plot_data = plot_data.sort_values(by=sort_by)

                f, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 10))


                sns.lineplot(x=x, y="CPU_WATTS", hue="CPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[1,0])
                sns.lineplot(x=x, y="GPU_WATTS", hue="GPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[1,1])


                if "gpu" in algo:
                    plot_data['PERF/J'] = plot_data['PERF']/plot_data['GPU_WATTS']
                    sns.lineplot(x=x, y="PERF",  hue="GPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[0,1],legend=False)
                    sns.lineplot(x=x, y='PERF/J',hue="GPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[2,1],legend=False)
                else:
                    plot_data['PERF/J'] = plot_data['PERF']/plot_data['CPU_WATTS']
                    sns.lineplot(x=x, y="PERF",  hue="CPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[0,0],legend=False)
                    sns.lineplot(x=x, y='PERF/J',hue="CPU_NAME", style=style,  data=plot_data, palette=self.arch_palette, markers=True,ax=axs[2,0],legend=False)
                    
                axs[0,1].set_ylabel("GPU "+plot_data['PERF_UNIT'].unique()[0])
                axs[1,1].set_ylabel("GPU Power (W)")
                axs[2,1].set_ylabel("FLOP/J")

                axs[0,0].set_ylabel("GPU "+plot_data['PERF_UNIT'].unique()[0])
                axs[1,0].set_ylabel("CPU Power (W)")
                axs[2,0].set_ylabel("FLOP/J")

                axs[2,0].semilogy()
                axs[2,1].semilogy()

                axs[2,0].set_xlabel("Size")
                axs[2,1].set_xlabel("Size")

                f.supxlabel(title)
                axs[1,0].legend(loc=2, prop={'size': 6}, bbox_to_anchor=[0.15, 2.6])
                axs[1,1].legend(loc=2, prop={'size': 6}, bbox_to_anchor=[0.15, 2.6])

                plt.subplots_adjust(wspace=0.22, hspace=0.05,bottom=0.1)
            
                plt.savefig("PPE_" + name + "_" + algo +".png",dpi=200,bbox_inches='tight')

    def GPU_TPE_plot(self,x,*args, **kwargs):
        hue = kwargs.get('hue', None)
        style = kwargs.get('style', None)
        sort_by = kwargs.get('sort_by', None)
        title = kwargs.get('title', None)

        plot_data = self.plot_data

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
        

