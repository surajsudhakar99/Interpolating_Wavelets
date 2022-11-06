from scipy import interpolate
import numpy as np
import time


# Class to generate Interpolating 1D wavelets
class Interp_Wlt_1D():
    def __init__(self):
        self.cutoff = 1e-5
        self.kind ='cubic'
        
    def interp_scl(self,xs,xe,j,k,support=6,level=4,return_func=False):
        support = support*(2**j)
        x = np.linspace(-support+k,support+k,int((2*support)/(2**j)+1))
        y = np.zeros((x.size,)) ; y[int((x.size-1)/2)] = 1
        
        for i in range(0,level):
            func = interpolate.interp1d(x,y,kind=self.kind,bounds_error=False,fill_value=0.0)
            if i==0:
                x = np.linspace(xs,xe,int(((xe-xs)/(2**(j-1)))+1))
                y = func(x) ; y[abs(y)<self.cutoff] = 0
            else:
                x = np.linspace(xs,xe,2*(x.size-1)+1)
                y = func(x) ; y[abs(y)<self.cutoff] = 0
                
        if return_func:
            return func
        else:
            x = np.linspace(xs,xe,2*(x.size-1)+1)
            y = func(x) ; y[abs(y)<self.cutoff] = 0
            return x, y
        
    def interp_scl_val(self,x_val,xs,xe,j,k,support=6,level=4):
        support = support*(2**j)
        x = np.linspace(-support+k,support+k,int((2*support)/(2**j)+1))
        y = np.zeros((x.size,)) ; y[int((x.size-1)/2)] = 1
        
        for i in range(0,level):
            func = interpolate.interp1d(x,y,kind=self.kind,bounds_error=False,fill_value=0.0)
            if i==0:
                x = np.linspace(xs,xe,int(((xe-xs)/(2**(j-1)))+1))
                y = func(x) ; y[abs(y)<self.cutoff] = 0
            else:
                x = np.linspace(xs,xe,2*(x.size-1)+1)
                y = func(x) ; y[abs(y)<self.cutoff] = 0
        
        y_val = func(x_val)
        if abs(y_val)<self.cutoff:
            return 0
        else:
            y_val
        
    
    def interp_wlt(self,xs,xe,j,k,support=6,level=4,return_func=False):
        if return_func==True:
            j = j-1 ; k = k+1
            support = support*(2**j)
            x = np.linspace(-support+k,support+k,int((2*support)/(2**j)+1))
            y = np.zeros((x.size,)) ; y[int((x.size-1)/2)] = -1

            for i in range(0,level):
                func = interpolate.interp1d(x,y,kind=self.kind,bounds_error=False,fill_value=0.0)
                if i==0:
                    x = np.linspace(xs,xe,int(((xe-xs)/(2**(j-1)))+1))
                    y = func(x) ; y[abs(y)<self.cutoff] = 0
                else:
                    x = np.linspace(xs,xe,2*(x.size-1)+1)
                    y = func(x) ; y[abs(y)<self.cutoff] = 0
                    
            return func
        else:
            x,y = self.interp_scl(xs,xe,j-1,k+1,support=6,level=4,return_func=False)
            y[abs(y)<self.cutoff] = 0.0
            return x,-y
    
    def inter_wlt_val(self,x_val,xs,xe,j,k,support=6,level=4):
        return -self.interp_scl_val(x_val,xs,xe,j-1,k+1)
    
    def diff_scl(self,xs,xe,j,k,support=6,level=4):
        func = self.interp_scl(xs,xe,j,k,support=6,level=4,return_func=True)
        x,_ = self.interp_scl(xs,xe,j,k,support=6,level=4,return_func=False)
        x[0] = x[0]+1e-6 ; x[x.size-1] = x[x.size-1]-1e-6
        
        g = derivative(func,x,dx=1e-6)
        g[abs(g)<self.cutoff] = 0
        
        return x,g
    
    def diff_wlt(self,xs,xe,j,k,support=6,level=4):
        func = self.interp_wlt(xs,xe,j,k,support=6,level=4,return_func=True)
        x,_ = self.interp_wlt(xs,xe,j,k,support=6,level=4,return_func=False)
        x[0] = x[0]+1e-6 ; x[x.size-1] = x[x.size-1]-1e-6
        
        g = derivative(func,x,dx=1e-6)
        g[abs(g)<self.cutoff] = 0
        
        return x,g       
        


#Class for 2D interpolating wavelets
class Interp_Wlt_2D(Interp_Wlt_1D):
    def __init__(self):
        super().__init__()
    
    def get_scl(self,xx,yy,j,loc,*args,**kwargs):
        k,l = loc
        x = xx[0,:] ; y = yy[:,0]
        
        func1 = self.interp_scl(x[0],x[x.size-1],j=j,k=k,return_func=True)
        func2 = self.interp_scl(y[0],y[y.size-1],j=j,k=l,return_func=True)
        
        fx = func1(x) ; fy = func2(y)
        zz = np.tensordot(fx,fy,axes=0)
        zz[abs(zz)<self.cutoff] = 0
        return zz
        
    def get_wlt(self,xx,yy,j,loc,nature='h',*args,**kwargs):
        k,l = loc
        x = xx[0,:] ; y = yy[:,0]
        
        func1 = self.interp_scl(x[0],x[x.size-1],j=j,k=k,return_func=True)
        func2 = self.interp_wlt(y[0],y[y.size-1],j=j,k=l,return_func=True)
        
        fx = func1(x) ; fy = func2(y)
        
        if nature=='h':
            zz = np.tensordot(fx,fy,axes=0)
        elif nature=='v':
            zz = np.tensordot(fy,fx,axes=0)
        elif nature=='d':
            zz = np.tensordot(fy,fy,axes=0)
        else:
            sys.exit('Wrong nature of Interpolating wavelet!')

        zz[abs(zz)<self.cutoff] = 0
        return zz


    def diff_dx(self,xx,yy,zz,n=1,order=3):
        # Uses forward and backward difference method at boundaries.
        # Central difference method is applied on the rest.
        if n==1:
            if order==3:
                x = xx[0,:] ; y = yy[:,0]
                diff_mat = np.zeros(zz.shape)
                diff_mat[:,0] = (zz[:,1]-zz[:,0])/(x[1]-x[0])  # Forward difference 
                for i in range(1,x.size-1):
                    diff_mat[:,i] = (zz[:,i+1]-zz[:,i-1])/(x[i+1]-x[i-1]) # Central difference
                diff_mat[:,x.size-1] = (zz[:,x.size-1]-zz[:,x.size-2])/(x[x.size-1]-x[x.size-2])  # Backward difference          
                return diff_mat
            else:
                sys.exit('Sorry, this is the only available order!')
        else:
            sys.exit('Only first derivative can be calculated as of now!')
            
    def diff_dy(self,xx,yy,zz,n=1,order=3):
        # Uses forward and backward difference method at boundaries.
        # Central difference method is applied on the rest.
        if n==1:
            if order==3:
                x = xx[0,:] ; y = yy[:,0]
                diff_mat = np.zeros(zz.shape)
                diff_mat[0,:] = (zz[1,:]-zz[0,:])/(y[1]-y[0])  # Forward difference
                for i in range(1,y.size-1):
                    diff_mat[i,:] = (zz[i+1,:]-zz[i-1,:])/(y[i+1]-y[i-1]) # Central difference
                diff_mat[y.size-1,:] = (zz[y.size-1,:]-zz[y.size-2,:])/(y[y.size-1]-y[y.size-2]) # Backward difference             
                return diff_mat
            else:
                sys.exit('Sorry, this is the only available order!')
        else:
            sys.exit('Only first derivative can be calculated as of now!')
        
    
    def get_scl_derivative(self,xx,yy,j,loc,partial,*args,**kwargs):

        zz = self.get_scl(xx,yy,j,loc)
        
        if partial=='dx':
            zz_diff = self.diff_dx(xx,yy,zz)
        elif partial=='dy':
            zz_diff = self.diff_dy(xx,yy,zz)
        else:
            sys.exit('Wrong "partial" derivative keyword')
        
        zz_diff[abs(zz_diff)<self.cutoff] = 0
        return zz_diff


    def get_wlt_derivative(self,xx,yy,j,loc,partial,*args,**kwargs):  
        
        nat = kwargs.get('nature')
        
        if (nat!=None) & (nat in ['h','v','d']):
            zz = self.get_wlt(xx,yy,j,loc,nature=nat)
            if partial=='dx':
                zz_diff = self.diff_dx(xx,yy,zz)
            elif partial=='dy':
                zz_diff = self.diff_dy(xx,yy,zz)
            else:
                sys.exit('Wrong "partial" derivative keyword')
        else:
            sys.exit('Incorrect nature of interpolating wavelet!')
        
        zz_diff[abs(zz_diff)<self.cutoff] = 0
        return zz_diff


#------------------------------------Numerical Differentiation Algorithms----------------------------------#
def derivative_fd(func,x,n=1,dx=1e-6):
    if n==1:
        return (func(x+dx)-func(x))/(dx)
    else:
        sys.exit('Only first derivative can be calculated as of now!')

def derivative_bd(func,x,n=1,dx=1e-6):
    if n==1:
        return (func(x)-func(x-dx))/(dx)
    else:
        sys.exit('Only first derivative can be calculated as of now!')

def derivative_cd(func,x,order=3,n=1,dx=1e-6):
    if n==1:
        if order==3:
            der = np.array([])
            for i in range(0,x.size):
                if i==0:
                    diff = derivative_fd(func,x[i])
                elif i==(x.size-1):
                    diff = derivative_bd(func,x[i])
                else:
                    diff = (func(x[i]+dx)-func(x[i]-dx))/(2*dx)
                der = np.append(der,diff)
        else:
            sys.exit('Sorry, this is the only available order!')
    else:
        sys.exit('Only first derivative can be calculated as of now!')
        
    return der
#-------------------------------------------------------------------------------------------------------------#

#------------------------------Generating Basis Function Matrix-----------------------------------------------#
# Gets the number of basis functions
def get_num_basis(xx,min_j,max_j):

    num_basis = 0
    x = xx[0,:]   
    for j in range(min_j,max_j+1):
        num_basis += (((x[x.size-1]-x[0])/(2**j))+1)**2   
    num_basis = 3*num_basis + (((x[x.size-1]-x[0])/(2**max_j))+1)**2  
    return int(num_basis)

# Places the basis functions on the grid
def get_func_on_grid(xx,yy,BFM,min_j,max_j,column,f,*args,**kwargs):
  
    pd = kwargs.get('partial')
    nat = kwargs.get('nature')
    
    x = xx[0,:] ; y = yy[:,0]
        
    for j in range(max_j,min_j-1,-1):
        no_div_x = int((x[x.size-1]-x[0])/(2**j)+1)
        no_div_y = int((y[y.size-1]-y[0])/(2**j)+1)
        xt = np.linspace(x[0],x[x.size-1],no_div_x)
        yt = np.linspace(y[0],y[x.size-1],no_div_y)
        for k in range(0,xt.size):
            for l in range(0,yt.size):
                BFM[:,column] = np.array(f(xx,yy,j=j,loc=(xt[k],yt[l]),partial=pd,nature=nat).flatten())
                column += 1  
    
    return column, BFM

# Sets up the Basis Function Matrix (BFM)
def get_BFM(xx,yy,min_j,max_j,*arg,**kwargs):
    
    func = kwargs.get('func')
    pd = kwargs.get('partial')
    
    if (func==None)|(len(func)!=2):
        func = [ip2d.get_scl,ip2d.get_wlt]
        print("'ip2d.get_scl' and 'ip2d.get_wlt' functions have been taken as default.")
    if pd==None:
        pd = 'dx' # Default partial derivative is wrt x
    
    x = xx[0,:] ; y = yy[:,0]
    
    num_basis = get_num_basis(xx,min_j,max_j)       
    W = np.zeros((xx.size,num_basis))
    col = 0
 
    ts = time.time()
    
    col, W = get_func_on_grid(xx,yy,W,max_j,max_j,column=col,partial=pd,f=func[0])
    col, W = get_func_on_grid(xx,yy,W,min_j,max_j,column=col,partial=pd,nature='v',f=func[1])
    col, W = get_func_on_grid(xx,yy,W,min_j,max_j,column=col,partial=pd,nature='h',f=func[1])
    col, W = get_func_on_grid(xx,yy,W,min_j,max_j,column=col,partial=pd,nature='d',f=func[1])
    
    te = time.time()
    
    print('--------------------------------------------------------------------------')
    print(f'Function: {func}')
    print(f'Number of basis functions = {col}')
    print(f'Matrix generation time = {te-ts} s')
    print('--------------------------------------------------------------------------')
    
    return W
#------------------------------------------------------------------------------------------------------------#
def rms_error(actual,pred):
    return np.sqrt((((actual-pred)**2).sum())/actual.size)
#------------------------------------------------------------------------------------------------------------#
