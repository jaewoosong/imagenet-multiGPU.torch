-- --------------------------------
-- Copyright (c) 2017, Jaewoo Song.
-- All rights reserved.
-- This code is forked from soumith/imagenet-multiGPU.torch
-- and follows its licence (which is just below).
-- ----------------------------------------------
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
-- -------------------------------------------------------------------------
-- 이 파일은 soumith/imagenet-multiGPU.torch 에서 포크해온 파일입니다.
-- 원본과 똑같은 라이센스가 사용됩니다.
-- ------------------------------------

require 'image'
paths.dofile('dataset.lua')
paths.dofile('util.lua')

-- 데이터를 불러오는 데에 사용되는 파일입니다.
-- 데이터를 불러오는 각각의 스레드에서 실행됩니다.
--------------------------------------------------

-- 학습 메타데이터에 대한 캐시 파일 (없으면 생성됨)
local trainCache   = paths.concat(opt.cache, 'trainCache.t7')
local testCache    = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

-- 데이터 디렉토리(opt.data)가 있는지 검사
if not os.execute('cd ' .. opt.data) then
    error(("디렉토리를 '%s' 으로 바꿀 수 없습니다."):format(opt.data))
end

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.cropSize, opt.cropSize}

local function loadImage(경로)
   -- 'double', 'float', 'byte' 사용 가능
   local 사진 = image.load(경로, 3, 'float')

   -- 사진 원본의 가로와 세로 
   local 가로 = 사진:size(3)
   local 세로 = 사진:size(2)

   -- 학습에 필요한 사진 크기
   local 가로변환 = loadSize[2]
   local 세로변환 = loadSize[3]

   -- 짧은 쪽을 찾아서 비율을 유지하면서 학습에 필요한 크기로 사진 변환
   if 가로 < 세로 then -- 위아래로 길쭉한 사진이면
      사진 = image.scale(사진, 가로변환, 세로변환 * 세로 / 가로)
   else -- 옆으로 길쭉한 사진이면
      사진 = image.scale(사진, 가로변환 * 가로 / 세로, 세로변환)
   end

   -- 결과값 반환
   return 사진
end

-- 각 채널의 평균과 표준편차. 계산하거나 혹은 코드의 뒷부분에서 디스크에서 불러옵니다.
local 평균, 표준편차

--------------------------------------------------------------------------------

--[[
    항목 1: 학습 데이터를 불러오는 기능을 만듭니다. (trainLoader)
            클래스 간 균형을 맞추어 데이터 셋 샘플링을 하고, 무작위 크롭도 합니다.
--]]

-- 사진을 불러오고 적당히 변형하는 함수 (무작위 크롭 등)
local trainHook = function(self, path)
   collectgarbage()
   local 입력사진 = loadImage(path) -- 위에서 만든 함수입니다.
   local 입력가로 = 입력사진:size(3)
   local 입력세로 = 입력사진:size(2)

   -- 무작위 크롭
   local 출력가로 = sampleSize[3]
   local 출력세로 = sampleSize[2]
   local 세로좌표1 = math.ceil(torch.uniform(1e-2, 입력세로-출력세로))
   local 가로좌표1 = math.ceil(torch.uniform(1e-2, 입력가로-출력가로))
   local 출력사진 = image.crop(입력사진, 가로좌표1, 세로좌표1,
                               가로좌표1 + 출력가로, 세로좌표1 + 출력세로)
   assert(출력사진:size(3) == 출력가로)
   assert(출력사진:size(2) == 출력세로)

   -- 0.5의 확률로 가로방향 뒤집기를 합니다.
   if torch.uniform() > 0.5 then 출력사진 = image.hflip(출력사진) end

   -- 평균, 표준편차
   for i=1,3 do -- 채널
      if 평균 then 출력사진[{{i},{},{}}]:add(-평균[i]) end
      if 표준편차 then 출력사진[{{i},{},{}}]:div(표준편차[i]) end
   end
   return 출력사진
end

if paths.filep(trainCache) then
   print('캐시에서 학습 메타데이터를 불러옵니다.')
   trainLoader = torch.load(trainCache) -- 중요한 변수
   trainLoader.sampleHookTrain = trainHook
   assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
          '캐시 파일의 내용이 opt.data와 경로가 다릅니다. '
             .. trainCache .. '에 있는 캐시를 지우고 프로그램을 다시 실행합니다.')
else
   print('학습 메타데이터를 생성합니다.')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- trainLoader가 정상적으로 동작하는지 확인
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "클래스 구조에 문제가 있습니다.")
   assert(class:min() >= 1, "클래스 구조에 문제가 있습니다.")

end

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
testHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if std then out[{{i},{},{}}]:div(std[i]) end
   end
   return out
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
   assert(testLoader.paths[1] == paths.concat(opt.data, 'val'),
          'cached files dont have the same path as opt.data. Remove your cached files at: '
             .. testCache .. ' and rerun the program')
else
   print('Creating test metadata')
   testLoader = dataLoader{
      paths = {paths.concat(opt.data, 'val')},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end
