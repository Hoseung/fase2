import numpy as np
import matplotlib.pyplot as plt 

class Seal_checker():
    def __init__(self, sec):
        self.sec = sec
    def check_decrypt(self, ctxt, n=20):
        """quickly check the value by decryption
        """
        print(self.sec.decrypt(ctxt)[:n])

    def check_ctxt(self, ctxt):
        """Check scale and "level" of the ciphertext
        note
        ----
        Both scale and level(modulus index) of operand ctexts must match.
        """
        print("Scale", np.log2(ctxt.scale()))
        print("chain index", self.sec.context.get_context_data(ctxt.parms_id()).chain_index())
    
    def plot_diff(self, org, ctxt, nh, nw, ch=3, stride=(1,1), 
                  vmin=None, vmax=None, unpad=0, out=False):
        """quickly plot an encrypted image and a plain image.

            parameters
            ----------
            org: 4-D tensor [batch, channel, height, width]
            ctxt: List, a n_channel-long list of 1-D ctxts, each containing a 2D image.
            nh: the "initial" size of the encrypted image 
            nw: the "initial" size of the encrypted image
            ch: channel index 
            stride: stride of ctxt
            vmin: minimum value of the diff map
            vmax: maximum value of the diff map
            unpad: if unpad > 0, plot only the inner region (to ignore erros from the wrong padding)
            
            The batch axis of ctxt is ignored (n_batch == 1 assumed)

            note
            ----
            strided convolution results in a sparse ctxt. 
        """
        if isinstance(stride, int):
            s_x = s_y = stride
        elif len(stride) == 2:
            s_x, s_y = stride

        x = self.sec.decrypt(ctxt[ch])[:nh*nw]

        #x = x.reshape(nh,nw)[s_x-1::s_x, s_y-1::s_y]
        x = x.reshape(nh,nw)[::s_x,::s_y]
        x2 = org[0,ch,...].detach().numpy()
        
        if unpad > 0:
            x = x[unpad:-unpad,unpad:-unpad]
            x2 = x2[unpad:-unpad,unpad:-unpad]

        diff_map = (x2-x)/ np.std(x2)
        fig, axs = plt.subplots(2,3, figsize=(9,5))
        axs[0,0].imshow(x2, vmin=x2.min(), vmax=x2.max())
        axs[0,0].set_title("target")
        axs[0,1].imshow(x, vmin=x2.min(), vmax=x2.max())
        axs[0,1].set_title("test")
        axs[1,0].imshow(x2-x)
        axs[1,0].set_title("diff")
        axs[1,1].imshow(diff_map)
        axs[1,1].set_title("diff/org.std")
        axs[0,2].hist(x.ravel(), histtype='step', lw=4, label='enc')
        axs[0,2].hist(x2.ravel(), histtype='step', lw=4, label='plain')
        axs[0,2].legend()
        print(f"average diff {np.mean(diff_map):.3f} sigma")

        plt.tight_layout()
        plt.show()

        if out: return x

        

class Heaan_checker():
    def __init__(self, hec):
        self.hec = hec
    def check_decrypt(self, ctxt, n=20):
        """quickly check the value by decryption
        """
        print(self.hec.decrypt(ctxt)[:n])

    def check_ctxt(self, ctxt):
        """Check scale and "level" of the ciphertext
        note
        ----
        Both scale and level(modulus index) of operand ctexts must match.
        """
        print("logp", ctxt.logp, "logq", ctxt.logq)
    
    def plot_diff(self, org, ctxt, nh, nw, ch=3, stride=(1,1), 
                  vmin=None, vmax=None, unpad=0, out=False):
        """quickly plot an encrypted image and a plain image.

            parameters
            ----------
            org: 4-D tensor [batch, channel, height, width]
            ctxt: List, a n_channel-long list of 1-D ctxts, each containing a 2D image.
            nh: the "initial" size of the encrypted image 
            nw: the "initial" size of the encrypted image
            ch: channel index 
            stride: stride of ctxt
            vmin: minimum value of the diff map
            vmax: maximum value of the diff map
            unpad: if unpad > 0, plot only the inner region (to ignore erros from the wrong padding)
            
            The batch axis of ctxt is ignored (n_batch == 1 assumed)

            note
            ----
            strided convolution results in a sparse ctxt. 
        """
        if isinstance(stride, int):
            s_x = s_y = stride
        elif len(stride) == 2:
            s_x, s_y = stride

        x = self.hec.decrypt(ctxt[ch])[:nh*nw]

        #x = x.reshape(nh,nw)[s_x-1::s_x, s_y-1::s_y]
        x = x.reshape(nh,nw)[::s_x,::s_y]
        x2 = org[0,ch,...].detach().numpy()
        
        if unpad > 0:
            x = x[unpad:-unpad,unpad:-unpad]
            x2 = x2[unpad:-unpad,unpad:-unpad]

        diff_map = (x2-x)/ np.std(x2)
        fig, axs = plt.subplots(2,3, figsize=(9,5))
        axs[0,0].imshow(x2, vmin=x2.min(), vmax=x2.max())
        axs[0,0].set_title("target")
        axs[0,1].imshow(x, vmin=x2.min(), vmax=x2.max())
        axs[0,1].set_title("test")
        axs[1,0].imshow(x2-x)
        axs[1,0].set_title("diff")
        axs[1,1].imshow(diff_map)
        axs[1,1].set_title("diff/org.std")
        axs[0,2].hist(x.ravel(), histtype='step', lw=4, label='enc')
        axs[0,2].hist(x2.ravel(), histtype='step', lw=4, label='plain')
        axs[0,2].legend()
        print(f"average diff {np.mean(diff_map):.3f} sigma")

        plt.tight_layout()
        plt.show()

        if out: return x