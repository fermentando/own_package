import os
import yt 
import glob
import sys
import numpy as np
from utils import *
import ffmpeg

class ImageConverter:

    def __init__(self, generateDir, saveDir, field='density'):
        self.generateDir = generateDir
        self.saveDir = saveDir
        self.field = fields[field]

        if not os.path.exists(self.saveDir): 
            os.makedirs(self.saveDir)

    def create(self, typefile='slice', mode="all", identifier=None):

        if mode == "single": identifier = identifier.zfill(3) if isinstance(identifier, str) else "*00"
        else: identifier = "*"
        print(f"out/*prim.{identifier}*.phdf")
        for filename in np.sort(glob.glob(os.path.join(self.generateDir, f"out/*{identifier}*.phdf"))):
            print(filename)
            index = int(filename.split("/")[-1].split(".")[2])
            if not os.path.exists(os.path.join(self.generateDir,f"{str.capitalize(typefile)}_{index}*")):
                print(f"Generating file number {index}. \n")
                ds = yt.load(filename)
                if str.lower(typefile) == 'slice': slc = yt.SlicePlot(ds, "z", self.field)
                else: slc = yt.ProjectionPlot(ds, "z", self.field, weight_field=('gas', 'mass'))
                slc.colorbar_location = 'left'
                slc.colorbar_width = 0.02
                slc.set_cmap(self.field, cmap='plasma_r')
                slc.set_zlim(self.field, 1e-26, 1e-24)
                slc.hide_axes()
                slc.save(os.path.join(self.saveDir, f"{str.capitalize(typefile)}_{index}.pdf"))


    def multiplot(self):
        
        tobeprojected = np.sort(glob.glob(os.path.join(self.generateDir, "out/*prim.[0-9]*.phdf")))
        for filename in tobeprojected:
            print(filename)
            index = int(filename.split("/")[-1].split(".")[2])
            if not os.path.exists(os.path.join(self.saveDir,f"multiplot_{index:03d}.png")):
                ds = yt.load(filename)

                p = yt.ProjectionPlot(ds, "z", fields.values(), weight_field=('gas', 'density'))
                p.set_log(("gas", "velocity_y"), False)

                fig = p.export_to_mpl_figure((2, 2))

                fig.tight_layout()
                fig.savefig(os.path.join(self.saveDir,f"multiplot_{index:03d}.png"))



        if len(tobeprojected) != 0:

            ffmpeg.input(os.path.join(self.saveDir,"multiplot_%03d.png"), framerate=2).output(
                os.path.join(self.saveDir,"multiplot.mp4"),
            ).run() 
        
    def hist(self):
        tobeprojected = np.sort(glob.glob(os.path.join(self.generateDir, "*prim.[0-9]*.phdf")))
        for filename in tobeprojected:
            print(filename)
            index = int(filename.split("/")[-1].split(".")[2])
            print(index)
            if not os.path.exists(os.path.join(self.saveDir,f"histT_{index:03d}.png")):
                
                ds = yt.load(filename)
                plt.hist(np.log10(ds.all_data()[('gas', 'temperature')]), log=True, bins = 50)
                plt.savefig(os.path.join(self.saveDir,f"histT_{index:03d}.png"))
                plt.clf()
        

        if len(tobeprojected) != 0:
            
            ffmpeg.input(os.path.join(self.saveDir,"histT_%03d.png"), framerate=2).output(
                os.path.join(self.saveDir,"histT_{}.mp4".format(self.generateDir.split("/")[-1])),
            ).run() 






if __name__ == "__main__":
    SIM_DIR = os.getcwd().split('/ferhi/')[-1]
    print(f"Main directory: {SIM_DIR}")
    sim = ImageConverter(os.path.join("/raven/ptmp/ferhi/", SIM_DIR), os.path.join(homeDir, "Figures" ,SIM_DIR))
    if len(sys.argv) == 1:
        sim.multiplot()
    else:
        sim.create(typefile=str(sys.argv[1]), mode='single', identifier=str(sys.argv[2]))
        #sim.create_movie()

