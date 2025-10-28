import os
import yt 
import glob
import sys
import numpy as np
import ffmpeg
import multiprocessing
import argparse
from utils import *
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter
from functools import partial


class ImageConverter:

    def __init__(self, generateDir, saveDir, field='density', num_workers=1):
        self.generateDir = generateDir
        self.saveDir = saveDir
        self.field = fields[field]
        self.num_workers = num_workers

        if not os.path.exists(self.saveDir): 
            os.makedirs(self.saveDir)

    def process_file(self, filename, typefile):
        """Processes a single file (slice or projection)"""
        index = int(filename.split("/")[-1].split(".")[2])
        output_path = os.path.join(self.saveDir, f"{str.capitalize(typefile)}_{index}.png")

        if not os.path.exists(output_path):
            print(f"Generating file number {index}. \n")
            ds = yt.load(filename)
            

            if str.lower(typefile) == 'slice':
                slc = yt.SlicePlot(ds, "z", self.field)
            else:
                slc = yt.ProjectionPlot(ds, "z", self.field, weight_field=('gas', 'mass'))

            # Set color limits based on field type
            if self.field == ('gas', 'density'):
                slc.set_zlim(self.field, 1e-26, 1e-22)
            elif self.field == ('gas', 'temperature'):
                slc.set_zlim(self.field, 1e4, 1e7)

            slc.colorbar_location = 'left'
            slc.colorbar_width = 0.02
            slc.set_cmap(self.field, cmap='plasma_r')
            slc.hide_axes()
            slc.save(output_path)

    def create(self, typefile='slice', mode="all", identifier=None):
        """Runs image generation in parallel"""

        identifier = identifier.zfill(3) if mode == "single" else "*"
        file_list = np.sort(glob.glob(os.path.join(self.generateDir, f"out/*{identifier}*.phdf")))

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.starmap(self.process_file, [(filename, typefile) for filename in file_list])

    def process_multiplot_one_field_only(self, filename, field):
        """Processes a single file for multiplot"""
        index = int(filename.split("/")[-1].split(".")[2])
        output_path = os.path.join(self.saveDir, f"multiplot_{str(field)}_{index:03d}.png")

        if not os.path.exists(output_path):
            ds = yt.load(filename)
            if False:
                left_edge, right_edge = ds.domain_left_edge, ds.domain_right_edge

                y_cut = left_edge[1] + 0.9 * (right_edge[1] - left_edge[1])
                new_center = [
                    0.5 * (left_edge[0] + right_edge[0]),
                    0.5 * (left_edge[1] + y_cut),
                    0.5 * (left_edge[2] + right_edge[2])
                ]

                region = ds.region(center=new_center,
                                left_edge=[left_edge[0], left_edge[1], left_edge[2]],
                        right_edge=[right_edge[0], y_cut, right_edge[2]])

            proj = yt.ProjectionPlot(ds, 'z', field, weight_field='density')#, data_source=region, center = new_center)

            proj.set_cmap(field, 'viridis')
            #proj.set_zlim(field, 1e-26, 1e-22)
            colorbar = proj.plots[field].cb

            colorbar.ax.set_aspect(40)  
            colorbar.ax.set_position([1.15, 0.1, 0.02, 0.8])  


            #proj.set_width((right_edge[0] - left_edge[0], y_cut - left_edge[1]))
            proj.save(output_path)
            del proj

    def process_multiplot(self, filename):
        """Processes a single file for multiplot"""
        index = int(filename.split("/")[-1].split(".")[2])
        output_path = os.path.join(self.saveDir, f"yt_movie_multiplot_{index:03d}.png")

        if not os.path.exists(output_path):
            ds = yt.load(filename)

            p = yt.ProjectionPlot(ds, "x", fields.values(), weight_field=('gas', 'density'))
            #p.set_log(("gas", "velocity_y"), False)
            #p.set_log(("gas", "velocity_x"), False)
            #p.set_log(("gas", "pressure_normalized"), False)
            #p.set_log(("gas", "shear"), False)

            # Apply colorbar limits based on field type
            for field in fields.values():
                if field == ('gas', 'density'):
                    p.set_zlim(field, 1e-26, 1e-22)
                    p.set_cmap(field, "viridis")
                elif field == ('gas', 'temperature'):
                    p.set_zlim(field, 1e4, 1e7)
                    p.set_cmap(field, "plasma")
                elif field == ("gas", "pressure_normalized"):
                    p.set_zlim(field, 1, 1e3)
                    p.set_log(field, True)
                    p.set_cmap(field, "coolwarm")
                elif field == ("gas", "velocity_y"):
                    p.set_cmap(field, "magma")
                    #p.set_zlim(field, 1e-2, 1)
                    p.set_unit(field, "km/s")
                elif field == ('gas', 'velocity_z'):
                    p.set_cmap(field, "cividis")
                    p.set_zlim(field, -2e5, 2e5)
                    p.set_unit(("gas", "velocity_z"), "km/s")
                elif field ==  ("gas", "mixing_gas_flag"):
                    p.set_zlim(field, 1e-2, 1)
                    p.set_cmap(field, "BuPu")


            #p.set_background_color("black")
            n_fields = len(p.fields)

            # Compute rows/cols to be roughly square
            cols = math.ceil(math.sqrt(n_fields))
            rows = math.ceil(n_fields / cols)

            fig = p.export_to_mpl_figure((rows, cols))
            for ax in fig.axes:
                if ax.get_images():  # Main plot axes
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel('y')
                    ax.set_ylabel('z')
                    
            for ax in fig.axes:
                for im in ax.get_images():
                    if im.colorbar:
                        im.colorbar.ax.yaxis.label.set_size(10)
                        im.colorbar.ax.tick_params(labelsize=10)
                        #im.colorbar.ax.yaxis.get_offset_text().set_visible(False)
                        #im.colorbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
                        
            
            fig.set_constrained_layout(True)
            fig.set_size_inches(16, 8)
            fig.savefig(output_path,bbox_inches='tight', pad_inches=0.05, dpi=300)
            print(f"Generated multiplot for file {index} at {output_path}")

    def multiplot(self, mode = "all"):
        """Runs multiplot generation in parallel"""
        file_list = np.sort(glob.glob(os.path.join(self.generateDir, "out/*prim.[0-9]*.phdf")))

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            if mode == "all":
                pool.map(self.process_multiplot, file_list)
            else:
                field = mode
                func = partial(self.process_multiplot_one_field_only, field=field)
                pool.map(func, file_list)


        if len(file_list) > 0:
            if mode == "all":
                ffmpeg.input(os.path.join(self.saveDir, "yt_movie_multiplot_%03d.png"), framerate=10).output(
                    os.path.join(self.saveDir, "yt_movie_multiplot.mp4")
                ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            else:
                field = mode
                files = glob.glob(os.path.join(self.saveDir, f"multiplot_{field}_*.png"))
                ffmpeg.input(os.path.join(self.saveDir, f"multiplot_{field}_%03d.png"), framerate=len(files)//30 if len(files)//30 > 2 else 2).output(
                    os.path.join(self.saveDir, f"multiplot_{field}.mp4")
                ).run(overwrite_output=True)#, capture_stdout=True, capture_stderr=True)

    def process_hist(self, filename):
        """Processes a single file for histogram"""
        index = int(filename.split("/")[-1].split(".")[2])
        output_path = os.path.join(self.saveDir, f"histT_{index:03d}.png")

        if not os.path.exists(output_path):
            ds = yt.load(filename)
            plt.hist(np.log10(ds.all_data()[('gas', 'temperature')]), log=True, bins=50)
            plt.savefig(output_path)
            plt.clf()

    def hist(self):
        """Runs histogram generation in parallel"""
        file_list = np.sort(glob.glob(os.path.join(self.generateDir, "*prim.[0-9]*.phdf")))

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.map(self.process_hist, file_list)

        if file_list:
            ffmpeg.input(os.path.join(self.saveDir, "histT_%03d.png"), framerate=2).output(
                os.path.join(self.saveDir, f"histT_{self.generateDir.split('/')[-1]}.mp4")
            ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)


if __name__ == "__main__":
    N_procs, user_args = get_n_procs_and_user_args()

    SIM_DIR = os.getcwd().split('/ferhi/')[-1]
    print(f"Main directory: {SIM_DIR}, Using {N_procs} processors.")

    sim = ImageConverter(
        os.path.join("/viper/ptmp/ferhi/", SIM_DIR),
        os.path.join(homeDir, "Figures", SIM_DIR),
        num_workers=N_procs
    )

    if not user_args: 
        print('Not user args') 
        sim.multiplot()
    elif 'density' in user_args:
        sim.multiplot(mode='density')
    elif 'temperature' in user_args:
        sim.multiplot(mode='temperature')
    else:
        print('Else')
        sim.multiplot(mode = str(sys.argv[1]))

