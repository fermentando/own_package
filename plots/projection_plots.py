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
        output_path = os.path.join(self.saveDir, f"{str.capitalize(typefile)}_{index}.pdf")

        if not os.path.exists(output_path):
            print(f"Generating file number {index}. \n")
            ds = yt.load(filename)

            if str.lower(typefile) == 'slice':
                slc = yt.SlicePlot(ds, "z", self.field)
            else:
                slc = yt.ProjectionPlot(ds, "z", self.field, weight_field=('gas', 'mass'))

            # Set color limits based on field type
            if self.field == ('gas', 'density'):
                slc.set_zlim(self.field, 1e-26, 1e-24)
            elif self.field == ('gas', 'temperature'):
                slc.set_zlim(self.field, 1e4, 1e6)

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

    def process_multiplot(self, filename):
        """Processes a single file for multiplot"""
        index = int(filename.split("/")[-1].split(".")[2])
        output_path = os.path.join(self.saveDir, f"multiplot_{index:03d}.png")

        if not os.path.exists(output_path):
            ds = yt.load(filename)
            p = yt.ProjectionPlot(ds, "z", fields.values(), weight_field=('gas', 'density'))
            p.set_log(("gas", "velocity_y"), False)

            # Apply colorbar limits based on field type
            for field in fields.values():
                if field == ('gas', 'density'):
                    p.set_zlim(field, 1e-26, 1e-24)
                elif field == ('gas', 'temperature'):
                    p.set_zlim(field, 1e4, 1e6)

            fig = p.export_to_mpl_figure((2, 2))
            fig.tight_layout()
            fig.savefig(output_path)

    def multiplot(self):
        """Runs multiplot generation in parallel"""
        file_list = np.sort(glob.glob(os.path.join(self.generateDir, "out/*prim.[0-9]*.phdf")))
        print(file_list[0])

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.map(self.process_multiplot, file_list)

        if len(file_list) > 0:
            ffmpeg.input(os.path.join(self.saveDir, "multiplot_%03d.png"), framerate=2).output(
                os.path.join(self.saveDir, "multiplot.mp4")
            ).run()

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
            ).run()

def get_n_procs():
    """Parse number of processors from command-line arguments"""
    parser = argparse.ArgumentParser(description="Set the number of processors.")
    parser.add_argument("--N_procs", type=int, default=1, help="Number of processors to use.")  # Change to optional
    args = parser.parse_args()
    return max(1, min(args.N_procs, multiprocessing.cpu_count()))  # Ensure it's in a valid range

if __name__ == "__main__":
    N_procs = get_n_procs()  # Read from command-line args
    SIM_DIR = os.getcwd().split('/ferhi/')[-1]
    print(f"Main directory: {SIM_DIR}, Using {N_procs} processors.")

    sim = ImageConverter(
        os.path.join("/raven/ptmp/ferhi/", SIM_DIR),
        os.path.join(homeDir, "Figures", SIM_DIR),
        num_workers=N_procs
    )

    user_args = []
    skip_next = False

    for arg in sys.argv[1:]:
        if skip_next:  
            skip_next = False  
            continue  
        if arg == "--N_procs":  
            skip_next = True  # Skip the next argument as well
            continue  
        user_args.append(arg)
    print("Arguments received:", sys.argv)
    print("user args: ", user_args)
    
    if not user_args: 
        print('Not user args') 
        sim.multiplot()
    else:
        print('Else')
        sim.create(typefile=str(sys.argv[1]), mode='single', identifier=str(sys.argv[2]))

