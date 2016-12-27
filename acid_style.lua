require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'

pi = 3.14159285359
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-start_number', 1, 'frame number to start at')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-init_image', 'examples/inputs/tubigen.jpg', 'initialization image')

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/train_val.prototxt')
cmd:option('-model_file', 'models/nin_imagenet_conv.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', '', 'layers for content')
cmd:option('-style_layers', 'relu0,relu1,relu2,relu3,relu5,relu6,relu7,relu8,relu9', 'layers for style')
cmd:option('-feedback_max', 1, 'level of image feedback')
cmd:option('-feedback_start', 0.1, 'level of feedback to start')
cmd:option('-feedback_growth', 2, 'rate that feedback increases')
cmd:option('-learning_max', 10, 'maximum learning rate after climb')
cmd:option('-learning_rate', 0.5)
cmd:option('-learning_growth', 2, 'rate that the learning rate increases')
cmd:option('-noise_max', 0.1, 'maximum noise ratio')
cmd:option('-noise_ratio', 0, 'ratio of noise to image feedback')
cmd:option('-noise_growth', 2, 'rate that the noise increases')
cmd:option('-warp_start',0, 'level of warping at start')
cmd:option('-warp_max', 2, 'level of warping at end (recommended value between 0 and 3)')
cmd:option('-warp_growth', 2, 'rate of warping growth')
cmd:option('-warp_image', '../inputs/warp2.png', 'warp basis')
cmd:option('-warp_ratio', 0.1, 'ratio of warp guide to self-warp')
cmd:option('-guide_image', '../inputs/warp2.png', 'guide basis')
cmd:option('-guide_weight', 0.6)
cmd:option('-warp_thresh', 0.5, 'threshold for warping')
cmd:option('-warp_breathe', 0.51, 'threshold for breathing instead of static warping')
cmd:option('-num_oscillations', 3, 'number of oscillations for the breathing')
cmd:option('-magmap_ratio', 0.7, 'how much warping is influenced by magnitude')
cmd:option('-rotational', false, 'rotational variation instead of breathing')

local function main(params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
  
  print(cnn:__tostring())
 
  local content_image = image.load(params.content_image, 3)
  local guide_image = image.load(params.guide_image, 3)
  local init_image = image.load(params.init_image, 3)
  local warp_image = image.load(params.warp_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  init_image = image.scale(init_image, params.image_size, 'bilinear')
  warp_image = image.scale(warp_image, content_image:size()[3], content_image:size()[2], 'bilinear')
  warp_image = warp_image:mul(2):add(-1)
  guide_image = image.scale(guide_image, content_image:size()[3], content_image:size()[2], 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  local guide_image_caffe = preprocess(guide_image):float()
  init_image = preprocess(init_image):float()
  -- preprocess guiding images to not fuck up
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end

  -- Handle style blending weights for multiple style inputs
  -- by using the neural style stuff, we can 'nudge' the visuals
  -- in the direction of the guide image
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end
  
  -- puts it on the gpu if requested (currently broken, needs fixing.)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      content_image_caffe = content_image_caffe:cuda()
      for i = 1, #style_images_caffe do
        style_images_caffe[i] = style_images_caffe[i]:cuda()
      end
    else
      content_image_caffe = content_image_caffe:cl()
      for i = 1, #style_images_caffe do
        style_images_caffe[i] = style_images_caffe[i]:cl()
      end
    end
  end
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting guide, serotonin and content loss modules
  -- additionally, we save the generated targets so we can fuck with them later.
  local content_losses, style_losses = {}, {}
  local serotonin_module, target_guide = {}, {}
  local content_guide = {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  local net2 = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
	net2:add(tv_mod)
  end
  wassup = style_images_caffe[1]:clone()
  local pewpew = 1
  
  net2 = nil
  collectgarbage()
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')

      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            avg_pool_layer:cuda()
          else
            avg_pool_layer:cl()
          end
        end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
		local guide = net:forward(guide_image_caffe):clone()
        local target = net:forward(content_image_caffe):clone()
		local guide_mean = guide:mean()
		local guide_std = guide:std()
		local target_mean = target:mean()
		local target_std = target:std()
		local mean = guide_mean+target_mean
		mean = mean*0.5
		local std = guide_std+target_std
		std = std*0.5
		target:add(-target_mean)
		guide:add(-guide_mean)
		target:div(target_std)
		guide:div(guide_std)
		target:mul(1-params.guide_weight):add(guide:mul(params.guide_weight))
		target:mul(std)
		target:add(mean)
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
		local noise_module = nn.Serotonin(params.noise_ratio):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
		--net:add(noise_module)
		table.insert(serotonin_module, noise_module)
		table.insert(content_guide, target)
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            gram = gram:cuda()
          else
            gram = gram:cl()
          end
        end
        local target = nil
        for i = 1, #style_images_caffe do
          local target_features = net:forward(style_images_caffe[i]):clone()
          local target_i = gram:forward(target_features):clone()
          target_i:div(target_features:nElement())
          --target_i:mul(style_blend_weights[i])
          if i == 1 then
            target = target_i
          else
            target:add(target_i)
          end
        end
		table.insert(target_guide, target)
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight/(math.sqrt(1+(i/1.3))), target, norm):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
	if torch.type(module) == 'nn.SpatialConvolution' then
		module.gradWeight = nil
		module.gradBias = nil
	end
  end
  collectgarbage()
  
  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float()--:mul(128)
  elseif params.init == 'image' then
    img = init_image:clone():float()
  elseif params.init == 'hybrid' then
	img2 = torch.randn(content_image:size()):float():mul(0.05)
	img = content_image_caffe:clone():float():mul(0.95):add(img2)
  elseif params.init == 'black' then
    img = content_image_caffe:clone():float():zero()
  else
    error('Invalid init type')
  end
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      img = img:cuda()
    else
      img = img:cl()
    end
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the loss modules and serotonin modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
	  learningRate = params.learining_rate,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  elseif params.optimizer == 'cg' then
	optim_state = {
	  maxIter = params.num_iterations,
	  learningRate = params.learining_rate,
      verbose=true,
	}
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
		print(img:size())
      local disp = deprocess(img:double()):clone()
	  print(disp:size())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(params.output_image, (t/params.save_iter)+params.start_number-1)
      if t == params.num_iterations then
        filename = params.output_image
      end
      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- we use ADAM because lbfgs is too smart for it's own good, and whines
  -- like a little babby over the fact that the feature maps don't match the gradients
  -- because of the serotonin layer. We don't need that shit anyway.
  local num_calls = 0
  -- we need the initialization values later, so we copy them into new appropriately named
  -- variables.
  local shitshagging = params.feedback_start
  local fuckfucking = params.noise_ratio
  local learnlearning = params.learning_rate
  local warpwarping = params.warp_start
  local step = 1/params.num_iterations
  local feedback_diff = params.feedback_max - params.feedback_start
  local serotonin_diff = params.noise_max - params.noise_ratio
  local learning_diff = params.learning_max - params.learning_rate
  local warp_diff = params.warp_max - params.warp_start  
  local function feval(x)
    num_calls = num_calls + 1 
		local should_wew = params.save_iter > 0 and num_calls % params.save_iter == 1
		if should_wew then
			-- get our warping tensor ready, mm baby yeah allocate that longstorage
			flow = torch.DoubleTensor()
			flow:resize(2,x:size()[2],x:size()[3])
			-- get a normalized version of the image for reasons. I don't have to explain shit.
			xnorm = img:clone():double()
			xnorm2 = warp_image:clone():double()
			for i = 1, 3 do
				mean = xnorm[i]:mean()
				std = xnorm[i]:std()
				xnorm[i] = xnorm[i]:add(-mean)
				xnorm[i] = xnorm[i]:div(std)
				xnorm[i] = xnorm[i]:div(xnorm[i]:max())
				mean = xnorm2[i]:mean()
				std = xnorm2[i]:std()
				xnorm2[i] = xnorm2[i]:add(-mean)
				xnorm2[i] = xnorm2[i]:div(std)
				xnorm2[i] = xnorm2[i]:div(xnorm[i]:max())
			end
			xnorm = xnorm:mul(1-params.warp_ratio):add(xnorm2:mul(params.warp_ratio))
			-- use this to change the orientation of the warp field over time
			rotation = ((1/params.num_iterations)*num_calls*pi*2*params.num_oscillations)
			-- build warping data based off gradient/contour information
			local magmap = 0
			warp_guide, magmap = build_warpmap(xnorm, rotation, false, true, params.warp_thresh, params.warp_breathe, params.magmap_ratio, params.rotational)
			print(warp_guide:min(), warp_guide:max())
			-- warp that motherfucker
			x = warpImage(x:double(), warp_guide:mul(warpwarping), params.noise_ratio):float()
			-- feedback between the input and output happens here
			x2 = content_image_caffe:clone():float()
			x = x:float():mul(shitshagging):add((x2:mul(1-shitshagging)))
			-- we need to reallocate img for reasons.
			img = x
			-- lin goes from 0 to 1 over the course of processing, useful for consistent
			-- modification of parameters
			lin = step * num_calls
			-- oooh yeah, that's one sexy easing function.
			easing = (1-((1-lin)*(1-lin)))
			-- adjust the various curves using exponents 
			feedback_smooth = easing^params.feedback_growth
			serotonin_smooth = easing^params.noise_growth
			learning_smooth = easing^params.learning_growth
			warp_smooth = easing^params.warp_growth
			-- use the curves to adjust parameters
			shitshagging = params.feedback_start + (feedback_diff * feedback_smooth)
			fuckfucking = params.noise_ratio + ( serotonin_diff * serotonin_smooth)
			optim_state.learningRate = params.learning_rate + (learning_diff * learning_smooth)
			warpwarping = params.warp_start + (warp_diff * warp_smooth)
		print(shitshagging)
		print(fuckfucking)
		print(optim_state.learningRate)
		print(warpwarping)
		end
	-- run the warped image through the net to add visualization data
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    local loss = 0
	-- collect losses of the base image and guide images
	local ayy = 1
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
	  -- while the feature map does get twiddled by serotonin, the losses don't.
	  -- as such we need to twiddle them on our own over here.
	  mod.target = torch.add(torch.mul(content_guide[ayy], 1-fuckfucking), torch.randn(content_guide[ayy]:size()):float():mul((content_guide[ayy]:max()-content_guide[ayy]:min())*fuckfucking))
	  ayy = ayy+1
    end
	ayy = 1
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
	  -- because of the same reasons, we need to twiddle the targets a bit.
	  mod.target = torch.add(torch.mul(target_guide[ayy], 1-fuckfucking), torch.randn(target_guide[ayy]:size()):float():mul((target_guide[ayy]:max()-target_guide[ayy]:min())*fuckfucking))	  ayy = ayy+1
    end
	-- and here we adjust the serotonin module strength.
	for _, mod in ipairs(serotonin_module) do
		mod.strength = fuckfucking
	end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- adam expects a vector for gradients, even if it's unused.
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('L-BFGS sucks, i advise switching to adam you dopey cunt')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM, as you should.')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  elseif params.optimizer == 'cg' then
    print('Running optimization with cg, totally experimental')
    for t = 1, params.num_iterations do
      local x, losses = optim.cg(feval, img, optim_state)
    end
  end
end


-- function to build a warpmap from the base image. It does some gradient/contour detection
-- and uses the info to build a warp field based on this.
-- todo: change it so we warp the feature maps instead of the image, more realistic.
function build_warpmap(input, rot, inv, lcn_, thresh, ratio, mag_ratio, rotational)
	hue = image.rgb2hsv(input)
	sat = image.rgb2hsv(input)
	hue = hue[1]:mul(2):add(-1):mul(pi)
	sat = sat[2]:mul(-2):add(1):mul(pi)
	-- make sure input is a greyscale image, i can't handle anything else ;_;
	input = image.rgb2y(input)
	local output = torch.Tensor(2,input:size()[2],input:size()[3]):zero()
	local dirmap = torch.Tensor(input:size()[2],input:size()[3]):zero()
	-- since we're doing edge detection, we use replication padding so the edge of the image doesn't count
	local padder = nn.SpatialReplicationPadding(1, 1, 1, 1)
	-- edge detection kernel. Horizontal kernel is inverted, because it looks nicer
	local vkern = torch.Tensor({{0.25,0.5,0.25},
					{0, 0, 0},
					{-0.25,-0.5,-0.25}})
	local hkern = torch.Tensor({{-0.25, 0, 0.25},
					{-0.5, 0, 0.5},
					{-0.25, 0, 0.25}})
	local forward = padder:forward(input)
	local vmap = torch.conv2(forward[1], vkern):add(1e-12)
	local hmap = torch.conv2(forward[1], hkern):add(1e-12)
	local magmap = torch.pow((torch.pow(hmap, 2):add(torch.pow(vmap,2))),0.5)
	-- using local contrast normalization keeps the flow consistent over the image,
	-- instead of concentrated on strong contrasts - our brain does this anyway.
	if lcn_ == true then
		padder = nn.SpatialReplicationPadding(4,4,4,4)
		-- image.lcn expects a 3d tensor
		local tempmag = torch.Tensor(1,magmap:size()[1],magmap:size()[2])
		tempmag[1] = magmap
		tempmag = padder:forward(tempmag)
		tempmag = image.lcn(tempmag)
		magmap = tempmag
		-- remove negative values that might crop up, because they'd fuck shit up
		magmap:add(-magmap:min())
	end
	--normalize the magnitude between 0 and 1 for consistency of warping
	magmap:div(magmap:max())
	-- inverting the magnitude can have interesting results, works best with LCN on
	if inv == true then
		magmap:mul(-1):add(1)
	end
	
	-- sadly torch.atan is trash, and math.atan doesn't behave as advertised.
	-- so instead i do this manually. Get your shit together guys.
	for i = 1, dirmap:size()[1] do
		for j = 1, dirmap:size()[2] do
			if vmap[i][j] > 0 then
				if hmap[i][j] > 0 then
					dirmap[i][j] = math.atan(vmap[i][j]/hmap[i][j])--+rot
				else
					dirmap[i][j] = math.atan(vmap[i][j]/hmap[i][j])+pi--+rot
				end
			else
				if hmap[i][j] > 0 then
					dirmap[i][j] = math.atan(vmap[i][j]/hmap[i][j])+(2*pi)--+rot
				else
					dirmap[i][j] = math.atan(vmap[i][j]/hmap[i][j])+pi--+rot
				end
			end
			if inv == false then
				if math.abs(magmap[i][j]) < thresh then
					magmap[i][j] = 0
				elseif math.abs(magmap[i][j]) > ratio then
					dirmap[i][j] = dirmap [i][j]+rot
				else
					dirmap[i][j] = dirmap [i][j]
				end
			else
				if math.abs(magmap[i][j]) < thresh then
					magmap[i][j] = 0
				elseif math.abs(magmap[i][j]) > ratio then
					dirmap[i][j] = dirmap [i][j]+rot
				else
					dirmap[i][j] = dirmap [i][j]
				end
			end
		end
	end
	dirmap:add(hue)
	dirmap:add(sat)
	-- reconstruct warp map using our fancy pants.
	magmap:mul(mag_ratio):add(1-mag_ratio)
	print(magmap:min(), magmap:max(), 'magmap')
	local warpy = torch.cmul(magmap, (torch.sin(dirmap)))
	local warpx = 0
	if rotational then
		warpx = torch.cmul(magmap, (torch.cos(dirmap)))
	else
		warpx = torch.cmul(magmap, (torch.sin(dirmap)))
	end
	local ysave = torch.add(warpy, -warpy:min())
	ysave:div(ysave:max())
	local xsave = torch.add(warpx, -warpx:min())
	xsave:div(xsave:max())
	local magsave = torch.add(magmap, -magmap:min())
	magsave:div(magsave:max())
	local dirsave = torch.add(dirmap, -dirmap:min())
	dirsave:div(dirsave:max())
	local vsave = torch.add(vmap, -vmap:min())
	vsave:div(vsave:max())
	local hsave = torch.add(hmap, -hmap:min())
	hsave:div(hsave:max())
	-- this is all for debugging, but they look pretty awesome.
	image.save('./ywarp.png', ysave)
	image.save('./xwarp.png', xsave)
	image.save('./mag.png', magsave)
	image.save('./dir.png', dirsave)
	image.save('./vmap.png', vsave)
	image.save('./hmap.png', hsave)
	output[1] = warpy
	output[2] = warpx
	return output, magmap
end
  
function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end

function build_filename2(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, 'orig_'..basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

local Serotonin, parent = torch.class('nn.Serotonin', 'nn.Module')

function Serotonin:__init(strength)
	parent.__init(self)
	self.strength = strength
	self.noise = torch.Tensor()
	self.mean = 0
	self.std = 0
	self.max = 0
	self.gradMax = 0
end

function Serotonin:updateOutput(input)
	-- to be blunt, all this does is add a noise bias to feature maps.
	-- Functionally identical to random activation due to psychedelic drugs,
	-- at least judging by how the result looks.
	self.output:resizeAs(input):copy(input)
	self.noise:resizeAs(input)
	self.noise = torch.randn(self.noise:size()):float()
	self.mean = self.output:mean()
	self.std = self.output:std()
	self.max = self.output:max()
	self.noise = self.noise:mul(self.std)
	self.noise = self.noise:add(self.mean)
	self.output = self.output:add(self.noise:mul(self.strength))
	self.output = self.output:div(self.output:max()):mul(self.max)
	return self.output
end	

-- we DON'T modify the gradients, because otherwise it negates the random activation.
function Serotonin:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	return self.gradInput
end

function Serotonin:accGradParameters(input, gradOutput)

end

function Serotonin:reset()
end

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.crit.sizeAverage = true
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  --else
   -- print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
	
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
  self.crit.sizeAverage = true
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- shamelessly stolen from artistic_video and tweaked for reasons.
function warpImage(img, flow, ratio)
  result = image.warp(img:double(), flow:double(), 'bilinear', true, 'pad', -1)
  local mean_pixel = torch.randn(result:size()):float()
  for x=1, result:size(2) do
    for y=1, result:size(3) do
      if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
        result[1][x][y] = mean_pixel[1][x][y]*ratio + img[1][x][y]*(1-ratio)
        result[2][x][y] = mean_pixel[2][x][y]*ratio + img[1][x][y]*(1-ratio)
        result[3][x][y] = mean_pixel[3][x][y]*ratio + img[1][x][y]*(1-ratio)
      end
    end
  end
  -- we clamp the output because otherwise it gets all fucky, especially with 
  -- lanzsoc filtering, which gives sharper results, but can also cause over/undershoot.
  result = torch.clamp(result, -128, 128)
  return result
end

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.rgb2yuv(content)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end

-- TV loss backward pass inspired by kaishengtai/neuralart
-- i'm not even going to pretend to know how this shit works.
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- oh shit son we're here, time to run this motherfucker.
local params = cmd:parse(arg)
main(params)
