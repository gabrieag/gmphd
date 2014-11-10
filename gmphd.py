
import math,numpy

from numpy import linalg,random

class filt():

    class hypot():

        def __init__(self,weight,mean,covar):

            self.weight=weight
            self.mean=mean
            self.covar=covar

        def div(self,*other):

            # Pre-compute the Cholesky factor of the
            # covariance, and its log-determinant.
            cholfact=linalg.cholesky(self.covar)
            logdet=numpy.log(cholfact.diagonal()).sum()

            div=[]

            # Evaluate the Kullback-Leibler divergence.
            for hypot in other:
                aux=linalg.cholesky(hypot.covar)
                div.append(numpy.sum(numpy.abs(linalg.solve(cholfact,aux))**2)/2.0
                           +numpy.sum(numpy.abs(linalg.solve(cholfact,self.mean-hypot.mean))**2)/2.0
                           +logdet-numpy.log(aux.diagonal()).sum()-float(self.mean.size)/2.0)

            return div

        def merge(self,*other):

            # Compute the weighted sums.
            self.mean*=self.weight
            self.covar*=self.weight
            for hypot in other:
                self.weight+=hypot.weight
                self.mean+=hypot.weight*hypot.mean
                self.covar+=hypot.weight*hypot.covar

            # Compute the mean and covariance
            # of the merged hypotheses.
            self.mean/=self.weight
            for hypot in other:
                diff=hypot.mean-self.mean
                self.covar+=hypot.weight*numpy.outer(diff,diff)
                self.covar/=self.weight

    def __init__(self,initmean,initcovar,transgain,transnoise,measgain,measnoise,clutterdens,
                 birthrate=0.0,clutterrate=0.0,survprob=1.0,detecprob=1.0):

        # Check the initial mean.
        if numpy.ndim(initmean)!=1:
            raise Exception('Initial mean must be a vector.')

        numstate=numpy.size(initmean)

        # Check the initial covariance.
        if numpy.ndim(initcovar)!=2 or numpy.shape(initcovar)!=(numstate,numstate):
            raise Exception('Initial covariance must be a {}-by-{} matrix.'.format(numstate,numstate))
        if not numpy.allclose(numpy.transpose(initcovar),initcovar):
            raise Exception('Initial covariance matrix must be symmetric.')
        try:
            cholfact=linalg.cholesky(initcovar)
        except linalg.LinAlgError:
            raise Exception('Initial covariance matrix must be positive-definite.')

        # Check the transition gain.
        if numpy.ndim(transgain)!=2 or numpy.shape(transgain)!=(numstate,numstate):
            raise Exception('Transition gain must be a {}-by-{} matrix.'.format(numstate,numstate))

        # Check the transition noise.
        if numpy.ndim(transnoise)!=2 or numpy.shape(transnoise)!=(numstate,numstate):
            raise Exception('Transition noise must be a {}-by-{} matrix.'.format(numstate,numstate))
        if not numpy.allclose(numpy.transpose(transnoise),transnoise):
            raise Exception('Transition noise matrix must be symmetric.')
        if numpy.any(linalg.eigvalsh(transnoise)<0.0):
            raise Exception('Transition noise matrix must be positive-semi-definite.')

        # Check the measurement gain.
        try:
            numobs,numcol=numpy.shape(measgain)
        except ValueError:
            raise Exception('Measurement gain must be a matrix.')
        if numcol!=numstate:
            raise Exception('Measurement gain matrix must have {} columns.'.format(numstate))

        # Check the measurement noise.
        if numpy.ndim(measnoise)!=2 or numpy.shape(measnoise)!=(numobs,numobs):
            raise Exception('Measurement noise must be a {}-by-{} matrix.'.format(numobs,numobs))
        if not numpy.allclose(numpy.transpose(measnoise),measnoise):
            raise Exception('Measurement noise matrix must be symmetric.')
        try:
            cholfact=linalg.cholesky(measnoise)
        except linalg.LinAlgError:
            raise Exception('Measurement noise matrix must be positive-definite.')

        # Check the clutter density.
        if not callable(clutterdens):
            raise Exception("Clutter density must be a callable function.")

        # Check the parameters.
        if not numpy.isscalar(birthrate) or birthrate<0.0:
            raise Exception("Birth rate must be a non-negative scalar.")
        if not numpy.isscalar(clutterrate) or clutterrate<0.0:
            raise Exception("Clutter rate must be a non-negative scalar.")
        if not numpy.isscalar(survprob) or survprob<0.0 or survprob>1.0:
            raise Exception("Survival probability must be a scalar between {} and {}.".format(0.0,1.0))
        if not numpy.isscalar(detecprob) or detecprob<0.0 or detecprob>1.0:
            raise Exception("Detection probability must be a scalar between {} and {}.".format(0.0,1.0))

        # Set the model.
        self.initmean=numpy.asarray(initmean)
        self.initcovar=numpy.asarray(initcovar)
        self.transgain=numpy.asarray(transgain)
        self.transnoise=numpy.asarray(transnoise)
        self.measgain=numpy.asarray(measgain)
        self.measnoise=numpy.asarray(measnoise)
        self.clutterdens=clutterdens

        # Set the parameters.
        self.birthrate=birthrate
        self.clutterrate=clutterrate
        self.survprob=survprob
        self.detecprob=detecprob

        self.__size__=numstate,numobs
        self.__hypot__=[]

    def __iter__(self):

        # Iterate over the hypotheses.
        for hypot in self.__hypot__:
            yield hypot.weight,hypot.mean,hypot.covar

    def pred(self):

        # Propagate the existing hypotheses.
        for i,hypot in enumerate(self.__hypot__):
            hypot.weight*=self.survprob
            hypot.mean=numpy.dot(self.transgain,hypot.mean)
            hypot.covar=numpy.dot(numpy.dot(self.transgain,hypot.covar),
                                  self.transgain.transpose())+self.transnoise

        # Create a new hypothesis.
        self.__hypot__.append(filt.hypot(self.birthrate,
                                         self.initmean.copy(),
                                         self.initcovar.copy()))

    def update(self,obs,tol=1.0e-9):

        numstate,numobs=self.__size__
        numhypot=len(self.__hypot__)

        # Check the observations.
        try:
            numrow,numpoint=numpy.shape(obs)
        except ValueError:
            raise Exception('Observations must be a matrix.')
        if numrow!=numobs:
            raise Exception('Observations matrix must have {} rows.'.format(numobs))

        # Allocate space for storing the log=likelihood of the
        # observations, and the conditional mean and covariance.
        loglik=numpy.zeros([numpoint,numhypot])
        mean=numpy.zeros([numstate,numpoint,numhypot])
        covar=numpy.zeros([numstate,numstate,numhypot])

        logconst=numobs*math.log(2.0*math.pi)/2.0

        for i,hypot in enumerate(self.__hypot__):

            weight=hypot.weight

            # Update the current hypothesis
            # assuming there is no detection.
            hypot.weight=1.0-self.detecprob

            # Compute the statistics of the innovation.
            innovmean=numpy.dot(self.measgain,hypot.mean)
            kalmgain=numpy.dot(hypot.covar,self.measgain.transpose())
            innovcovar=numpy.dot(self.measgain,kalmgain)+self.measnoise

            cholfact=numpy.linalg.cholesky(innovcovar)

            # Evaluate the log-likelihood of the observations.
            loglik[:,i]=math.log(self.detecprob*weight)-logconst-numpy.log(cholfact.diagonal()).sum()\
                -numpy.sum(numpy.abs(linalg.solve(cholfact,obs-innovmean[:,numpy.newaxis]))**2,axis=0)/2.0

            # Construct the Kalman and Joseph gain matrices.
            kalmgain=linalg.solve(innovcovar,kalmgain.transpose()).transpose()
            josgain=numpy.eye(numstate)-kalmgain.dot(self.measgain)

            # Compute the conditional mean and covariance.
            mean[:,:,i]=numpy.dot(josgain,hypot.mean[:,numpy.newaxis])+numpy.dot(kalmgain,obs)
            covar[:,:,i]=numpy.dot(josgain,numpy.dot(hypot.covar,josgain.transpose()))+\
                numpy.dot(kalmgain,numpy.dot(self.measnoise,kalmgain.transpose()))

        # Compute the log-scale factors.
        logscale=loglik.max(axis=1)
        logscale+=numpy.log(numpy.sum(numpy.exp(loglik-logscale[:,numpy.newaxis]),axis=1))

        for j in range(numpoint):

            loginten=numpy.log(self.clutterrate*self.clutterdens(obs[:,j]))

            # Update the log-scale factor.
            logscale[j]=max(logscale[j],loginten)+math.log1p(math.exp(-abs(logscale[j]-loginten)))

            # Add the hypotheses that
            # have significant weight.
            for i in range(numhypot):
                weight=math.exp(loglik[j,i]-logscale[j])
                if weight>tol:
                    self.__hypot__.append(filt.hypot(weight,mean[:,j,i],covar[:,:,i]))

    def prune(self,truncthres=1.0e-2,mergethres=1.0,maxhypot=100):

        __hypot__=[]

        # Sort the hypotheses according to their weight.
        self.__hypot__.sort(key=lambda x:x.weight)

        scale=sum(hypot.weight for hypot in self.__hypot__)

        # Iteratively merge the hypotheses.
        while self.__hypot__:
            hypot=self.__hypot__.pop()
            if hypot.weight>truncthres:

                mergeind=[]
                remind=[]

                # Find all the hypotheses that are close, in terms
                # of information, to that with the largest weight.
                for j,div in enumerate(hypot.div(*self.__hypot__)):
                    (mergeind if div<mergethres else remind).append(j)

                # Merge them into a single hypothesis.
                hypot.merge(*[self.__hypot__[j] for j in mergeind])
                self.__hypot__=[self.__hypot__[j] for j in remind]

                __hypot__.append(hypot)

        # Sort the hypotheses according to their weight.
        __hypot__.sort(key=lambda x:x.weight,reverse=True)
        __hypot__=__hypot__[:maxhypot]

        if __hypot__:

            scale/=sum(hypot.weight for hypot in __hypot__)

            # Scale the hypotheses.
            for hypot in __hypot__:
                hypot.weight*=scale

        self.__hypot__=__hypot__
