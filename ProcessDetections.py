
import collections,math,numpy,os,scipy

from matplotlib import image,patches,pyplot
from numpy import linalg,random
from os import path
from scipy import special

import gmphd

relpath='flir_17_Sept_2013/ETHZ-ASL'
imwidth=324
imheight=256
sampfreq=20.0

numdim=2
maxpos=500.0
maxvel=10.0
accelnoise=5.0
obsnoise=10.0
birthrate=0.01
clutterrate=1.0
survprob=0.99
detecprob=0.95
truncthres=1.0e-12
mergethres=0.5
maxhypot=20

names=collections.defaultdict(list)
detections=collections.defaultdict(list)

abspath=path.join(path.dirname(__file__),relpath)

# Store the image names.
for file in os.listdir(path.join(abspath,'8bit')):
    if file.endswith('.png'):
        name,extension=file.split('.')
        names[int(name)]=file

# Load the detections.
with open(path.join(abspath,'detection.txt'),mode='r') as file:
    for line in file:
        frame,col,row,width,height=map(int,line.split(',')[:-1])
        detections[frame].append((col,row,width,height))

if numdim==2:

    # Create the initial mean and covariance.
    initmean=numpy.array([float(imwidth)/2.0,float(imheight)/2.0,0.0,0.0])
    initcovar=numpy.diag(numpy.array([maxpos,maxpos,maxvel,maxvel])**2)

    timestep=1.0/sampfreq

    # Create the transition gain and noise.
    transgain=numpy.kron(numpy.array([(1.0,timestep),(0.0,1.0)]),numpy.eye(numdim))
    transnoise=numpy.kron(accelnoise**2*numpy.array([(timestep**2/4.0,timestep/2.0),
                                                     (timestep/2.0,1.0)]),numpy.eye(numdim))

    # Create the measurement gain and noise.
    measgain=numpy.kron(numpy.array([1.0,0.0]),numpy.eye(numdim))
    measnoise=numpy.kron(numpy.array([obsnoise**2]),numpy.eye(numdim))

# Define the clutter density.
area=imwidth*imheight
clutterdens=lambda x:1.0/float(area)

# Instantiate the filter.
filt=gmphd.filt(initmean,initcovar,transgain,transnoise,measgain,measnoise,clutterdens,
                birthrate,clutterrate,survprob,detecprob)

scale=special.gammaincinv(numdim,0.99)

# Create a figure and a pair of axes.
fig,axes=pyplot.subplots()
axes.invert_yaxis()

for frame in range(min(names.keys()),max(names.keys())):
    try:

        # Perform a prediction-update step.
        filt.pred()
        if frame in detections:
            obs=numpy.array(detections[frame],dtype=float).transpose()
            filt.update(obs[:numdim,:],numpy.spacing(1.0))
        filt.prune(truncthres=truncthres,
                   mergethres=mergethres,
                   maxhypot=maxhypot)

        axes.cla()

        # Plot the image.
        plot=axes.imshow(image.imread(path.join(abspath,'8bit',names[frame])))
        plot.set_cmap('hot')

        for weight,mean,covar in filt:

            # Decompose the dispersion matrix of the position.
            eigval,eigvec=linalg.eigh(covar[numpy.ix_([0,1],[0,1])])

            width,height=numpy.sqrt(eigval)
            angle=numpy.degrees(numpy.arctan2(*eigvec[:,0][::-1]))

            # Create an ellipse depicting the position uncertainty.
            ellip=patches.Ellipse(xy=mean[numpy.ix_([0,1])],
                                  width=scale*width,
                                  height=scale*height,
                                  angle=angle,
                                  alpha=min(max(weight,0.0),1.0),
                                  facecolor='blue',
                                  edgecolor='none')

            axes.add_artist(ellip)

        # Plot the detections.
        if frame in detections:
            for x,y,u,v in detections[frame]:
                axes.plot([x,x+u,x+u,x,x],[y,y,y+v,y+v,y],
                          color='black',
                          linewidth=2.0)

        axes.set_xlim(-0.5,float(imwidth)-0.5)
        axes.set_ylim(float(imheight)-0.5,-0.5)

        # Indicate the frame number.
        axes.annotate('Frame {}'.format(frame),
                      xy=(0.99,0.99),
                      xycoords='axes fraction',
                      fontsize=16,
                      horizontalalignment='right',
                      verticalalignment='top')

        fig.show()
        pyplot.pause(0.01)

    except KeyboardInterrupt:
        break
