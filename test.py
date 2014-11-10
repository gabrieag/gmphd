
import math,numpy,scipy

from matplotlib import patches,pyplot
from numpy import linalg,random
from scipy import special

import gmphd

numdim=2
poslim=-500.0,500.0
timestep=1.0
maxspeed=100.0
conflevel=0.99

initnoise=10.0
transnoise=5.0
measnoise=10.0

birthrate=0.1
clutterrate=50.0
survprob=0.99
detecprob=0.99

# Create the initial mean and covariance.
initmean=numpy.kron(numpy.array([0.0,0.0]),numpy.ones(numdim))
initcovar=numpy.kron(initnoise**2*numpy.diag([1.0,1.0/4.0]),numpy.eye(numdim))

# Create the transition gain and noise.
transgain=numpy.kron(numpy.array([(1.0,timestep),(0.0,1.0)]),numpy.eye(numdim))
transnoise=numpy.kron(transnoise**2*numpy.array([(timestep**2/4.0,timestep/2.0),
                                                 (timestep/2.0,1.0)]),numpy.eye(numdim))

# Create the measurement gain and noise.
measgain=numpy.kron(numpy.array([1.0,0.0]),numpy.eye(numdim))
measnoise=numpy.kron(numpy.array([measnoise**2]),numpy.eye(numdim))

area=abs(max(poslim)-min(poslim))**2

# Define the clutter density.
def clutter(obs):
    if numpy.all(obs>min(poslim)) and numpy.all(obs<max(poslim)):
        return 1.0/area
    else:
        return 0.0

# Instantiate the filter.
filt=gmphd.filt(initmean,initcovar,transgain,transnoise,measgain,measnoise,clutter,
                birthrate,clutterrate,survprob,detecprob)

target={0:random.multivariate_normal(initmean,initcovar)}

# Create a figure and a pair of axes.
fig,axes=pyplot.subplots()

scale=special.gammaincinv(numdim,conflevel)

while True:

    ind=[]

    # Simulate existing targets.
    for i in target.keys():
        if random.rand()<survprob:
            target[i]=random.multivariate_normal(numpy.dot(transgain,target[i]),transnoise)
            pos,vel=numpy.split(target[i],2)
            if numpy.all(pos>min(poslim)) and numpy.all(pos<max(poslim)):
                vel*=min(1.0,maxspeed/max(linalg.norm(vel),maxspeed))
                target[i]=numpy.concatenate([pos,vel])
                ind.append(i)

    target={i:target[i] for i in ind}

    # Simulate new targets.
    for i in range(random.poisson(birthrate)):
        target[len(target)]=random.multivariate_normal(initmean,initcovar)

    obs=[]

    # Simulate target measurements.
    for i in target.keys():
        if random.rand()<detecprob:
            obs.append(random.multivariate_normal(numpy.dot(measgain,target[i]),measnoise))

    # Simulate clutter measurements.
    for i in range(random.poisson(clutterrate)):
        obs.append(numpy.array([random.uniform(*poslim),
                                random.uniform(*poslim)]))

    # Run the filter for a single
    # prediction-update step.
    filt.pred()
    filt.update(numpy.array(obs).transpose())
    filt.prune(maxhypot=10)

    axes.cla()

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

    if target:

        # Plot the targets.
        x,y,u,v=zip(*target.values())
        axes.quiver(x,y,u,v,
                    color='red',
                    scale_units='xy',
                    angles='xy',
                    scale=10.0/36.0,
                    zorder=100)

    if obs:

        # Plot the observations.
        x,y=zip(*obs)
        axes.plot(x,y,
                  color='black',
                  marker='x',
                  markersize=5.0,
                  linestyle='none')

    axes.set_xlim(*poslim)
    axes.set_ylim(*poslim)

    fig.show()
    pyplot.pause(0.25)
