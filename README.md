# Acid Style

This is a modification of jcjohnson's neural style, changed to make accurate psychedelic animations. More coming soon, but for now you'll have to put up with my shit.

Requires a copy of NIN-imagenet, although it should be trivial to modify it to accept other networks. NIN gives good results however, and is pretty light on the ol' ram

#USEAGE:
first, generate a basis map using this shit here. Note that gpu should always be disabled until i get off my lazy arse and port it to opencl. Someone else can do cuda if they want to, nvidia is literally satan.

th acid_style.lua -normalize_gradients -print_iter 3 -save_iter 3 -style_image \<input image>  -gpu -1 -content_image \<input image> -init_image \<input image> -optimizer adam -tv_weight 3e-4 -init random -feedback_max 1 -feedback_start 1 -noise_ratio 0.5 -noise_max 0 -learning_max 8 -learning_rate 16 -style_scale 1 -style_weight 200 -image_size 1024 -num_iterations 900 -content_weight 0 -warp_start 3 -warp_max 0 -warp_growth 2 -warp_image \<input image> -feedback_growth 1 -guide_image \<input image> -guide_weight 0 -warp_thresh 0 -warp_breathe 0.01 -warp_ratio 0 -num_oscillations 3 -rotational -magmap_ratio 0 -output_image \<output image> 

Congrats, now you have a trippy looking image! but that's only half of the story... now it's time to use that basis image to trippify shit. Using your generated image on the SAME input image (seriously, it makes a huge difference which basis you choose), run this shit

th acid_style.lua -normalize_gradients -print_iter 3 -save_iter 3 -style_image \<previously generated image>  -gpu -1 -content_image \<input image> -init_image \<input image> -optimizer adam -tv_weight 5e-4 -init hybrid -feedback_max 0.9 -feedback_start 0.9 -noise_ratio 0.05 -noise_max 0.05 -learning_max 4 -learning_rate 4 -style_scale 1 -style_weight 200 -image_size 1024 -output_image \<output image> -num_iterations 1800 -content_weight 5 -warp_start 1.5 -warp_max 1.5 -warp_growth 2 -warp_image \<input image> -feedback_growth 1 -guide_image \<input image> -guide_weight 0 -warp_thresh 0.3 -warp_breathe 0.501 -warp_ratio 0 -num_oscillations 6 -rotational

If you want to go in more depth with what the parameters do, for now just play with them or read the code - i'll put up proper descriptions when i'm less lazy.

#TROUBLESHOOTING
> it's too blurry!

try turning down the feedback start/max, and possibly either turn down or turn up warp start/warp max.
  
> it's too trippy!

turn down the learning rate, noise ratio/max

> too much is moving!

increase the warp thresh parameter.

> too little is moving!

decrease the warp thresh, if that doesn't help increase the warp ratio a little bit. PROTIP - anything in the params you see that isn't above 1, is not meant to go above 1. please no breaky.

> it's moving too slow

increase the num_oscillations parameter

> it's moving too fast!

reverse the above.

> why the hell is the number of images controlled by num iterations divided by save iter?

i'm lazy, i'll change it later aight?

  
#EXAMPLES:

https://gfycat.com/WeightyDaringHoatzin
https://gfycat.com/ThirdThoroughFruitfly
https://gfycat.com/LonelyPoisedKob
http://i.imgur.com/e9N4c59.gifv
https://i.imgur.com/CQReIsd.gifv
https://gfycat.com/FrequentWickedKusimanse
https://gfycat.com/FantasticViciousIndigowingedparrot
https://gfycat.com/MindlessUnselfishArgentinehornedfrog



original code based off
```
@misc{Johnson2015,
  author = {Johnson, Justin},
  title = {neural-style},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jcjohnson/neural-style}},
}
```

Bunch of warp stuff ripped from
```
@TechReport{RuderDB2016,
  author = {Manuel Ruder and Alexey Dosovitskiy and Thomas Brox},
  title = {Artistic style transfer for videos},
  institution  = "arXiv:1604.08610",
  year         = "2016",
}
```

pretty sure that's all in terms of attribution, but i'll go over it with a fine tooth comb later to make sure everyone gets their citations.

