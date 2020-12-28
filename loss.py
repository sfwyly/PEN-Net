@tf.function()
def train_step(generator,discriminator, image_list, mask_image_list, mask_list):  # image为原图 0 - 1.0
    with tf.GradientTape() as tape,tf.GradientTape() as dis_tape:
        B, H, W, C = image_list.shape
        image_list = tf.cast(image_list, dtype=tf.float32)
        mask_image_list = tf.cast(mask_image_list, dtype=tf.float32)
        mask_list = tf.cast(mask_list, dtype=tf.float32)
        pyramid_feats,gen_image_list = generator([mask_image_list, mask_list])
        gen_image_list = tf.cast(gen_image_list, dtype=tf.float32)
        
        dis_real_feat = discriminator(image_list)
        dis_fake_feat = discriminator(gen_image_list)
        dis_loss = (dis_real_feat + dis_fake_feat)/2.
        
        for feat in pyramid_feats:
            pyramid_loss += l1_loss(feat,tf.image.resize(images,feat.shape[1:3],"bilinear"))
        
        hole_loss = l1_loss(gen_image_list*mask_list, image_list*mask_list)/tf.reduce_mean(mask_list)
        valid_loss = l1_loss(gen_image_list*(1-mask_list), image_list*(1-mask_list))/tf.reduce_mean(1-mask_list)
        adv_feats = discriminator(gen_image_list)
        adversarial_loss = calv(adv_feats,1)
        gen_loss = hole_loss * 6. + valid_loss * 1. + pyramid_loss * 0.5 + adversarial_loss * 0.1
        
    generator_grads = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    dis_grads = dis_tape.gradient(gen_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(dis_grads, discriminator.trainable_variables))
    
    return gen_loss,dis_loss
