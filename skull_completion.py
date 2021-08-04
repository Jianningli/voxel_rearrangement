from __future__ import division
import os
import time
import sys
from glob import glob
from conv3 import *
import numpy as np
import nrrd
from loader import *
from pre_post_processing import *
from scipy.ndimage import zoom

class auto_encoder(object):
    def __init__(self, sess):
        self.sess           = sess
        self.phase          = 'train'
        self.batch_size     = 1
        self.inputI_size    = 128
        self.inputI_chn     = 1
        self.output_chn     = 2
        self.lr             = 0.0001
        self.beta1          = 0.3
        self.epoch          = 10000
        self.model_name     = 'auto_encoder.model'
        self.save_intval    = 100
        self.outputI_size   = 128
        self.build_model()

        self.chkpoint_dir   = "../ckpt" 
        self.train_data_dir = "../train"
        self.test_data_dir = "../test"
        self.testing_results_dir="../test_save/"
        self.testing_defected_dir="../test_defect"

        self.save_implant_dir = "../implant/"
        self.train_label_dir = "../complete_skull"





     # weighted 3D voxel-wise  dice loss function
    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, 2)
        dice = 0
        for i in range(2):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice


    def dice4valid(self,pred,input_gt):
        dice_c = []
        for c in range(self.output_chn):
            ints = np.sum(((input_gt[0,:,:,:]==c)*1)*((pred[0,:,:,:]==c)*1))
            union = np.sum(((input_gt[0,:,:,:]==c)*1) + ((pred[0,:,:,:]==c)*1)) + 0.0001
            dice_c.append((2.0*ints)/union)
        return dice_c



    def build_model(self):
        print('building patch based model...')       
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,self.inputI_size,self.inputI_size,64, self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,self.inputI_size,self.inputI_size,64,1], name='target')
        self.soft_prob , self.task0_label = self.encoder_decoder(self.input_I)
        #3D voxel-wise dice loss
        self.main_dice_loss = self.dice_loss_fun(self.soft_prob, self.input_gt[:,:,:,:,0])
        #self.main_softmax_loss=self.softmax_crossentropy_loss(self.soft_prob, self.input_gt[:,:,:,:,0])
        # final total loss
        self.dice_loss=200000000*self.main_dice_loss
        self.Loss = self.dice_loss
        # create model saver
        self.saver = tf.train.Saver()
        self.saver_pretrained=tf.train.Saver()



    def encoder_decoder(self, inputI):
        phase_flag = (self.phase=='train')
        concat_dim = 4 
        #inputI (1, 256, 256, 128,1)
        conv1_1 = conv3d(input=inputI, output_chn=64, kernel_size=5, stride=2, use_bias=True, name='conv1')
        conv1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv1_batch_norm")
        conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')
        print('1',conv1_relu.shape)
        conv2_1 = conv3d(input=conv1_relu, output_chn=128, kernel_size=5, stride=2, use_bias=True, name='conv2')
        conv2_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv2_batch_norm")
        conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')
        print('2',conv2_relu.shape)
        conv3_1 = conv3d(input=conv2_relu, output_chn= 256, kernel_size=5, stride=2, use_bias=True, name='conv3a')
        conv3_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_1_batch_norm")
        conv3_relu = tf.nn.relu(conv3_bn, name='conv3_1_relu')
        print('3',conv3_relu.shape)
        conv4_1 = conv3d(input=conv3_relu, output_chn=512, kernel_size=5, stride=2, use_bias=True, name='conv4a')
        conv4_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_1_batch_norm")
        conv4_relu = tf.nn.relu(conv4_bn, name='conv4_1_relu')
        print('4',conv4_relu.shape)
        conv5_1 = conv3d(input=conv4_relu, output_chn=512, kernel_size=5, stride=1, use_bias=True, name='conv5a')
        conv5_bn = tf.contrib.layers.batch_norm(conv5_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv5_1_batch_norm")
        conv5_relu = tf.nn.relu(conv5_bn, name='conv5_1_relu')
        print('5',conv5_relu.shape)
        feature= conv_bn_relu(input=conv5_relu, output_chn=256, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='conv6_1')
        print('feature',feature.shape)
        deconv1_1 = deconv_bn_relu(input=feature, output_chn=256, is_training=phase_flag, name='deconv1_1')
        deconv1_2 = conv_bn_relu(input=deconv1_1, output_chn=128, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_2')
        print('6',deconv1_2.shape)
        deconv2_1 = deconv_bn_relu(input=deconv1_2, output_chn=128, is_training=phase_flag, name='deconv2_1')
        deconv2_2 = conv_bn_relu(input=deconv2_1, output_chn=64, kernel_size=5,stride=1, use_bias=True, is_training=phase_flag, name='deconv2_2')
        print('7',deconv2_2.shape)
        deconv3_1 = deconv_bn_relu(input=deconv2_2, output_chn=64, is_training=phase_flag, name='deconv3_1')
        deconv3_2 = conv_bn_relu(input=deconv3_1, output_chn=64, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='deconv3_2')
        print('8',deconv3_2.shape)
        deconv4_1 = deconv_bn_relu(input=deconv3_2, output_chn=32, is_training=phase_flag, name='deconv4_1')
        deconv4_2 = conv_bn_relu(input=deconv4_1, output_chn=32, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='deconv4_2')
        print('9',deconv4_2.shape)
        pred_prob1 = conv_bn_relu(input=deconv4_2, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
        pred_prob = conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, name='pred_prob')
        pred_prob2 = conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, name='pred_prob2')
        pred_prob3 = conv3d(input=pred_prob2, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, name='pred_prob3')
        

        print('10',pred_prob.shape)
        soft_prob=tf.nn.softmax(pred_prob3,name='task_0')
        print('11',soft_prob.shape)
        #11 (1, 128, 128, 128,2)
        task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
        print('12',task0_label.shape)
        #12 (1, 128, 128, 128)
        return  soft_prob,task0_label




    def train(self):
        print('training online model')
        total_time=[]
        print('initializing graph...')
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        loss_summary_0 =tf.summary.scalar('dice loss',self.Loss)
        #loading pretrained model
        if self.restore_chkpoint(self.chkpoint_dir):
            print('************start training from last saved checkpoint*********')
        else:
            print('**********training from scratch**********************')
        #self.saver_pretrained.restore(self.sess, tf.train.latest_checkpoint('D:/skull-volume/checkpoint_folder/real_batch_model_ckpt/'))
        self.sess.run(init_op)
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        print('initializing done...')
        counter=1

        data_list =glob('{}/*.nrrd'.format(self.train_data_dir))
        label_list=glob('{}/*.nrrd'.format(self.train_label_dir))

        i=0
        # save loss value for each epoch in loss.txt
        loss_log = open("loss.txt", "w")
        for epoch in np.arange(self.epoch):
            i=i+1
            start_time = time.time()
            print('creating batches for training epoch :',i)
            batch_img1, batch_label1= load_random_patch_pair(data_list,label_list)

            #nrrd.write('data_defected.nrrd',batch_img1[0,:,:,:,0],hd)
            #nrrd.write('label_.nrrd',batch_label1[0,:,:,:,0],hl)

            print('epoch:',i )
            _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: batch_img1, self.input_gt: batch_label1})
            train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: batch_img1})
            print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(batch_label1),np.sum(train_output0)))
            summary_0=self.sess.run(loss_summary_0,feed_dict={self.input_I: batch_img1,self.input_gt: batch_label1})
            self.log_writer.add_summary(summary_0, counter)           
            loss_log.write("%s\n" % (cur_train_loss))
            print('current training loss:',cur_train_loss)
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)
        loss_log.close()




    def test(self):
        print('testing patch based model...')  
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load_chkpoint(self.chkpoint_dir):
            print(" *****Successfully load the checkpoint**********")
        else:
            print("*******Fail to load the checkpoint***************")

        pair_list=glob('{}/*.nrrd'.format(self.test_data_dir))
        defected_list=glob('{}/*.nrrd'.format(self.testing_defected_dir))
        for i in range(len(pair_list)):

            test_input,header,zz=load_random_test_pair(pair_list,i)
            defected,header=nrrd.read(defected_list[i])

            start_time = time.time()

            reconstructed=np.zeros(shape=(512,512,zz))
            zl=zz//64
            patch_idx=0
            t1=time.time()
            for x in range(4):
                for y in range(4):
                    for z in range(zl):
                        test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input[patch_idx]})
                        reconstructed[x*128:(x+1)*128,y*128:(y+1)*128,z*64:(z+1)*64]=test_output[0,:,:,:]
                        patch_idx=patch_idx+1

            for x in range(4):
                for y in range(4):
                    test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input[patch_idx]})
                    reconstructed[x*128:(x+1)*128,y*128:(y+1)*128,zz-64:zz]=test_output[0,:,:,:]
                    patch_idx=patch_idx+1


            t2=time.time()
            print('reconstruction time:',t2-t1)

            reconstructed_post_processed=post_processing(reconstructed)
            implant=reconstructed_post_processed-defected
            implants_post_processed=pre_processing(implant)

      
            filename1=self.testing_results_dir+"skull%d.nrrd"%i
            filename2=self.save_implant_dir+"implants%d.nrrd"%i
            nrrd.write(filename1,reconstructed_post_processed,header)
            nrrd.write(filename2,implants_post_processed,header)




    # saving checkpoint 
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s" % ('superres')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)




    # loading checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % ('superres')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



    # loading checkpoint file
    def restore_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % ('superres')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False





