#include "enqueue_ofccl_dev.h"

// TODO: nccl最新的代码里，这部分的设计和实现都变了。
// 
// Copy src to dst and fill extra size with zeroes
// 这个是保证在一次调用复制完最多512B，并且以16B为单位。
// 这个不要求src dst同一类型
template<typename Tdst, typename Tsrc>
static __device__ void copyToShmemOneShot(Tdst *dst, Tsrc const *src, int tid, int nthreads) { // nccl的这个的函数签名里有个nthreads参数，但是并没有用，应该是为了和下边那个作区分，现在我们可以区分开了，反而带上nthreads是区分不开的。
  static_assert(sizeof(Tdst)%(2*sizeof(uint64_t )) == 0 && sizeof(Tsrc)%(2*sizeof(uint64_t)) == 0,
      "copyToShmemOneShot needs sizes which are multiple of 16B");
  static_assert(sizeof(Tdst) >= sizeof(Tsrc), "Tdst size is too small");
  static_assert(sizeof(Tdst) <= WARP_SIZE*2*sizeof(uint64_t), "copyToShmemOneShot limited to 512B to make sure it can always be done in one cycle");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  uint64_t *shmemPtr = shmemCvtPtr_ofccl(d); // 由于这个地方，这个函数只能用于dst是shmem的情况了。
  int offset = 2*tid;
  uint64_t v0, v1;
  if (offset >= sizeof(Tsrc)/sizeof(uint64_t)) {
    v0 = v1 = 0ULL;
  } else {
    v0 = s[offset] ; v1 = s[offset+1];
  }
  if (offset < sizeof(Tdst)/sizeof(uint64_t)) storeShmem128_ofccl(shmemPtr+offset, v0, v1);
}

// 这个可以直接用到任意一轮搞不完的数据结构的复制吧。
// 这个要求src dst同一类型。
// turn的作用：   
template<typename T>
static __device__ int copyToShmemLoop(T *dst, T const *src, int tid, int nthreads, int turn=0) {
  static_assert(sizeof(uint64_t) <= alignof(T), "Uhoh");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  int t = tid - turn;
  if (t < 0) t += nthreads;
  int n = sizeof(T)/sizeof(uint64_t); // n 代表要复制的数据结构包含了几个8Byte

  int delta = (n + WARP_SIZE-1) & -WARP_SIZE; // round up to warp lane 0; 要把n和WARP_SIZE处理对齐了。
  //  32 = 0000 0000 0010 0000
  // -32 = 1111 1111 1110 0000，低位不变，高位都置1。大一的东西忘却了。。
  // 所以delta相当于n相对于32的“向上取整”，即向上取到32的整数倍。

  if (delta < nthreads) { // 总的要传的 8Byte 的个数小于blockDim.x（我们的case里是thrdLimit）
    turn += delta;
    if (turn >= nthreads) turn -= nthreads; // 在第一次调用里这个不会成立，应该是为了后续的调用使用
  }
  else
    turn = 0; // 如果总的要传的 8Byte 的个数超过了blockDim.x，那就不用管turn了。所以turn就是为了雨露均沾，让所有线程都干活

  n -= t; // 对每个线程来说，砍掉比tid小的几项，不用自己管。
  d += t; // 对每个线程来说，自己从tid的偏移量开始管。
  s += t;
  #pragma unroll // 指示要循环展开。
  for (int i=0; i < divUp(sizeof(T), WARP_SIZE*sizeof(uint64_t)); i++) {
    if (n > 0) {
      *d = *s;
      d += nthreads;
      s += nthreads;
      n -= nthreads; // “一轮”完成 nthreads个8 Byte的复制。
    }
  }
  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> hello", sharedCollCtx.rank, blockIdx.x, tid);
  return turn;
}

// 这个的目的应该是在“切片并行复制”之后，恢复标量的语义
// 但是没用，而且在buffer里的数据是0.5，或者其他数字时，导致卡住。log发现buffer里的数字是0.25，可以正常运行，并且没有进入这里。所以直接注释了吧。
// TODO: 但是这里卡住，总还是怪怪的。有空看看吧。
// static __device__ void ofcclRedopPtrDeref(struct ncclWorkElem* we) {
//   if (we->header.type != ncclWorkTypeUnused && we->redOpArgIsPtr) {
//     /* redOpArg is a pointer to the scalar value, so we'll dereference it
//      * here so that redOpArg holds the bits of the scalar going forward.
//      * The tricky thing is we don't know its type T since that's encoded in
//      * the funcIndex. Because it would be difficult to get sizeof(T) from
//      * funcIndex, we'll cheat and just dereference the largest possible size
//      * given the alignment of the pointer. We might be reading in more bytes
//      * than we need but that's harmless.
//      */
//     if (we->redOpArg%2 != 0)
//       we->redOpArg = *reinterpret_cast<uint8_t*>(we->redOpArg);
//     else if (we->redOpArg%4 != 0)
//       we->redOpArg = *reinterpret_cast<uint16_t*>(we->redOpArg);
//     else if (we->redOpArg%8 != 0)
//       we->redOpArg = *reinterpret_cast<uint32_t*>(we->redOpArg);
//     else
//       we->redOpArg = *reinterpret_cast<uint64_t*>(we->redOpArg);
//     // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, we->redOpArgIsPtr = %d, we->redOpArg = %llu", sharedCollCtx.rank, blockIdx.x, threadIdx.x, we->redOpArgIsPtr, we->redOpArg);
//   }
// }

// share mem用超了。
// TODO: 可以不同的algo、proto使用不同的数据类型，不过可以看看是不是有意义
__shared__ CollCtx sharedCollCtx; // 不能static，primitives要用

__shared__ BlkStatus blkStatus; // 取消static，放到prim里边打印log。
// TODO: 下边这几个可以尝试用constant，先不急
static __shared__ int sharedBlkCount4Coll[MAX_LENGTH];
static __shared__ int sharedThrdCount4Coll[MAX_LENGTH];

static __device__ int sqRead(SQ *sq, SQE *target, int thrdCudaDev) {

  unsigned long long int currSqFrontier = blkStatus.sqReadFrontier;
  
  OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, enter, sqReadFrontier = %llu, sq->head=%llu, sq->tail=%llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, currSqFrontier), DevLogicSqHeadInline(sq), DevLogicSqTailInline(sq)); // sharedCollCtx.rank是在loadCtx之后才有效的，在此之前想打印sqRead的情况，需要使用thrdCudaDev，不然会搞出乌龙。

  if (DevLogicSqTailInline(sq) == DevRingBufferLogicFrontier(sq, currSqFrontier)) {
    return -1;
  }
  // 先读过来，然后再判断，最后更新状态：sqe->counter; 以及在恰当的时候commit read
  *target = *DevRingBufferGetFrontier(sq, currSqFrontier);
  if (target->quit) {
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Get quit", thrdCudaDev, bid, threadIdx.x);
    return 0;
  }

  int oldCounter = atomicAdd(&(DevRingBufferGetFrontier(sq, currSqFrontier)->counter), 1); // 将自己读了的sqe的counter加1，代表有人读过了，有一个block不需要再读这个sqe了，后来再有人读这个的时候加完了去判断。

  blkStatus.sqReadFrontier++; // 这次读到了，那对于当前这个block来说，下一个可读的位置前进一个。

  OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, update counter = %d for collId = %d, @ %llu", thrdCudaDev, blockIdx.x, threadIdx.x, oldCounter + 1, DevRingBufferGetFrontier(sq, currSqFrontier)->collId, DevRingBufferLogicFrontier(sq, currSqFrontier));

  __threadfence(); // 保证device上的各个block不要乱序看到。

  unsigned long long int sqHead;
  if (oldCounter + 1 == gridDim.x) {
    do {
      sqHead = atomicCAS(&sq->head, currSqFrontier, currSqFrontier + 1);
    } while (sqHead != currSqFrontier);

    OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, update sq->head, blkStatus.sqReadFrontier = %llu, sq->head = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, blkStatus.sqReadFrontier), DevLogicSqHeadInline(sq));
  }
  
  return 0;
}

static __device__ int cqWrite(CQ *cq, CQE *cqe, int thrdCudaDev, unsigned long long int *cqeWriteCnt) {
  if (CqFull(cq)) {
    // not an error; caller keeps trying.
    return -1;
  }

  unsigned long long int myCqFrontier = atomicAdd(&(cq->frontier), 1); // 占坑，我就往这里写了，用的是old值，新的cq->tail预期是atomicAdd之后的cq->frontier，也就是myCqFrontier + 1。
  // 两个线程同时调用atomicAdd，是严格保证各自返回的。

  // *(blkStatus.collCounters + 5 + cqe->collId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) = DevRingBufferLogicFrontier(cq, myCqFrontier);
  // *(blkStatus.collCounters + 6 + cqe->collId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) = cq->tail;

  __threadfence();

  DevRingBufferGetFrontier(cq, myCqFrontier)->collId = cqe->collId; // 那这里也应该各自写进去了。

  __threadfence_system();

  // atomicCAS返回地址上的old值，是否修改体现不在返回值上。
  unsigned long long int cqTail;
  do {
    cqTail = atomicCAS(&cq->tail, myCqFrontier, myCqFrontier + 1);
  } while(cqTail != myCqFrontier); // while这里是观察CAS里的条件是否被满足，如果观察到这个条件满足了，那也就可以确定Swap的操作也就完成了。

  // *(blkStatus.collCounters + 1 + cqe->collId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;
  OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, put %lluth CQE for collId = %d @ %llu and update cq->tail", thrdCudaDev, blockIdx.x, threadIdx.x, ++(*cqeWriteCnt), cqe->collId, DevRingBufferLogicFrontier(cq, myCqFrontier));

  return 0;
}

// TODO: 为了性能，考虑恢复成多线程一起复制的写法。
static __device__ void copyNcclWorkElem (struct ncclWorkElem &dstElem, const struct ncclWorkElem &srcElem) {
  dstElem.header.funcIndex = srcElem.header.funcIndex;
  dstElem.header.type = srcElem.header.type;
  dstElem.header.nWarps = srcElem.header.nWarps;
  dstElem.header.isLast = srcElem.header.isLast;

  dstElem.regUsed = srcElem.regUsed;
  dstElem.direct = srcElem.direct;
  dstElem.redOpArgIsPtr = srcElem.redOpArgIsPtr;
  dstElem.sendbuff = srcElem.sendbuff;
  dstElem.recvbuff = srcElem.recvbuff;
  dstElem.count = srcElem.count;
  dstElem.lastChunkSize = srcElem.lastChunkSize;
  dstElem.root = srcElem.root;
  dstElem.bid = srcElem.bid;
  dstElem.nChannels = srcElem.nChannels;
  dstElem.redOpArg = srcElem.redOpArg;
}

static __device__ int initContexts(int thrdCudaDev, int collCount, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int turn) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // int nthreads = blockDim.x;
  // 构建任务列表
  // TODO: 并行提高复制效率。
  if (tid == 0) {
    for (int i = 0; i < collCount; i++) {
      int collId = globalCollIds[i];
      // 以下这两个变量会限制很多行为。
      int blkLimit = sharedBlkCount4Coll[collId] = globalBlkCount4Coll[collId];
      sharedThrdCount4Coll[collId] = globalThrdCount4Coll[collId];

      // 下边这部分逻辑在在blkStatus.hasVolunteerQuitted == 1的情况下不执行，曾经退出过，恢复的时候就不要重新初始化了。
      if (blkStatus.hasVolunteerQuitted == 0) {
        // 每个block一份globalShmem
        CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;

        // ***** 移植ncclKernel的逻辑 *****
        if (bid < blkLimit) {
          // ncclDevComm *comm = globalDevComm7WorkElems[collId].comm;
          // turn = copyToShmemLoop(&(globalCollCtx4Blk7Coll->comm), comm, tid, nthreads, turn);
          // // 一个奇技淫巧：get address of channel without incurring indirect load from ncclDevComm::channels
          // // 这里通过bid选择了合适的channel，很多集合通信真正执行时用到的硬件信息就存在channel里边。
          // ncclChannel *channel = &((ncclDevCommAndChannels*)comm)->channels[bid];
          // turn = copyToShmemLoop(&(globalCollCtx4Blk7Coll->channel), channel, tid, nthreads, turn); // 尝试使用oneshot，会报错warp misaligned，所以看来必须用loop。

          // // nccl中限制只在bid=0里进行这样的拷贝，对于ofccl而言，ofcclShmem就是任务列表，所以对于所有的线程，我们都把同样的work存进去；
          // turn = copyToShmemLoop(&(globalCollCtx4Blk7Coll->work.elems[0]), &(globalDevComm7WorkElems[collId].first), tid, nthreads, turn); // nccl 2.12里边这地方用copyToShmemOneShot进行拷贝，但是oneShot的实现使用了与shared mem相关的内联汇编，所以这里也使用loop进行拷贝。
          // // nccl中接下来要处理channel.workFifoDev，然而对于目前的ofccl，只处理first就好，channel.workFifoDev不会有其他任务了。
          // __syncthreads(); // 等待全部线程加载完成

          // if (globalCollCtx4Blk7Coll->work.header.type == ncclWorkTypeColl) {
          //   // #define NCCL_MAX_WORK_ELEMENTS (NCCL_WORK_SIZE / sizeof(struct ncclWorkElem))=512/64=8
          //   // 原来这个写法，应该是想修改we->redOpArg，不过修改we->redOpArg一个线程就够了，所以让理论上最多的线程来工作，咱们保留就好。
          //   if (tid < NCCL_MAX_WORK_ELEMENTS) ofcclRedopPtrDeref(&(globalCollCtx4Blk7Coll->work.elems[tid]));
          // } // 目前不用考虑其他ncclWorkType
          // __syncthreads();
        
          /* ****** 手动加载用得到的shmemData ****** */
          ncclDevComm *comm = globalDevComm7WorkElems[collId].comm;
          ncclChannel *channel = &((ncclDevCommAndChannels*)comm)->channels[bid];

          globalCollCtx4Blk7Coll->ringPrev = channel->ring.prev;
          globalCollCtx4Blk7Coll->ringNext = channel->ring.next;
          globalCollCtx4Blk7Coll->ringIndex = channel->ring.index;
          globalCollCtx4Blk7Coll->devPeers = channel->devPeers; // 直接赋值指针

          globalCollCtx4Blk7Coll->rank = comm->rank;
          globalCollCtx4Blk7Coll->nRanks = comm->nRanks;
          globalCollCtx4Blk7Coll->abortFlag = comm->abortFlag;

          for (int i = 0; i < NCCL_NUM_PROTOCOLS; i++) {
            globalCollCtx4Blk7Coll->buffSizes[i] = comm->buffSizes[i];
          }

          copyNcclWorkElem(globalCollCtx4Blk7Coll->workElem, globalDevComm7WorkElems[collId].first);

          /* ****** 上下文 ****** */
          globalCollCtx4Blk7Coll->executing = 0; 
          // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, collId=%d, excuting = %d", thrdCudaDev, blockIdx.x, threadIdx.x, collId, globalCollCtx4Blk7Coll->executing);
          // globalCollCtx4Blk7Coll->numDoneThrds = 0;

          globalCollCtx4Blk7Coll->sqeReadCnt = 0;
          globalCollCtx4Blk7Coll->cqeWriteCnt = 0;
          
          // bugfix: 下边原来都是设置的globalBlk2CollId2CollCtx->XXXX，相当于都设置了第0个block的第0个coll。。。。。。。
          globalCollCtx4Blk7Coll->saveCtx7Quit = 0;
          globalCollCtx4Blk7Coll->loadAgain = 0;
          globalCollCtx4Blk7Coll->slice4SimpleGenericOp = 0;
          globalCollCtx4Blk7Coll->offset4SimpleGenericOp = 0;

          globalCollCtx4Blk7Coll->currentStep4RingAllReduce = 0;
          globalCollCtx4Blk7Coll->gridOffset4RingAllReduce = 0;
        }
      }
    }
  }
  ofcclBarrier(1);
  return turn;
}

// 为了初步实现按需启停，增加一个“空read计数，读不到新的，增加计数”
static __device__ void checkSQ7TidyTaskQ(int thrdCudaDev, SQ *sq, CollCtx *globalBlk2CollId2CollCtx, int *failCnt, int *finallyQuit) {
  int bid = blockIdx.x;
  
  SQE target;

  // 能读到，假如是正常SQE，把信息在任务列表里记录一下；假如是quit，那也记录一下
  // 读不到新东西那就算了
  
  if (sqRead(sq, &target, thrdCudaDev) == -1) {
    *failCnt += 1;
    if (blkStatus.numActiveColls > 0) {
      *failCnt = 0; // TODO: 更改failCnt的更新逻辑，觉得自己死锁了，虽然任务列表不空，但是半天动不了，也可以退。

      // bugfix：这次没读到新的，但是taskQ不为空，这时候要把里边的空项删除。
      int new_numActiveColls = 0;
      for (int i = 0; i < blkStatus.numActiveColls; ++i) {
        CollCtx *globalCollCtx4Blk7OldActiveColl = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + blkStatus.activeCollIds[i];
        if (globalCollCtx4Blk7OldActiveColl->executing == 1) {
          // 在同一个数组上就地操作。new_numActiveColls一定是<=i的，所以不会有问题。
          blkStatus.activeCollIds[new_numActiveColls++] = blkStatus.activeCollIds[i];
        }
      }
      blkStatus.numActiveColls = new_numActiveColls;
    }
    // if (bid == 0) { // 所有sqe 0号block一定可以读到，0号读失败的时候，说明当前用户提交的sqe，至少都被0号block看到了，需要及时更新sq->head，让block 1的工作通过block 0确认，并且最终被block 1看到。
    //   if (ensureSqHeadUpToDate(sq, blkStatus.sqReadFrontier)) {
    //     // 如果成功更新了sq->head，那么应该重置failCnt，延缓0号block的主动退出。
    //     *failCnt = 0;
    //   }
    // }
    return;
  } else {
    // TODO: 更改failCnt的更新逻辑，觉得自己死锁了，虽然任务列表不空，但是半天动不了，也可以退。
    *failCnt = 0;
    if (target.quit) {
      blkStatus.quit = 1;
      // if (bid == 0) {
        *finallyQuit = 1; // TODO: 为了最后每个block都保证打印统计信息，挺不优雅的
      // }
      return;
    }

    // 正常读到了SQE的话，需要往global的globalBlk2CollId2CollCtx表项里边写入，更新blkStatus.numActiveColls
    int newActiveCollId = target.collId;
    int blkLimit = sharedBlkCount4Coll[newActiveCollId]; // 需要参与新读到的coll的block才会进行后续操作。
    if (bid < blkLimit) {
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + newActiveCollId;
      if (globalCollCtx4Blk7Coll->executing == 1) {
        OFCCL_LOG(OFCCL_FATAL, "Rank<%d> Blk<%d> Thrd<%d> globalCollCtx4Blk7Coll->executing should be 0! sq->head = %llu, sq->tail = %llu, blkStatus.sqReadFrontier = %llu", thrdCudaDev, bid, threadIdx.x, DevLogicSqHeadInline(sq), DevLogicSqTailInline(sq), DevRingBufferLogicFrontier(sq, blkStatus.sqReadFrontier));
      }
      // TODO: 可以考虑一下这个地方加入原子操作，保证没有重入的风险。重入指一个正在执行的集合通信又被提起请求。
      // 虽然这里是操作globalMemory，但是我们设计的是各个block自己的数据结构自己操作。具体操作的都是每个block的0号线程，所以应该不会有啥问题。
      globalCollCtx4Blk7Coll->executing = 1;
      OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, read %lluth SQE for collId = %d, sq->head = %llu, sq->tail = %llu, blkStatus.sqReadFrontier = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->sqeReadCnt), newActiveCollId, DevLogicSqHeadInline(sq), DevLogicSqTailInline(sq), DevRingBufferLogicFrontier(sq, blkStatus.sqReadFrontier));
      globalCollCtx4Blk7Coll->workElem.sendbuff = target.sendbuff;
      globalCollCtx4Blk7Coll->workElem.recvbuff = target.recvbuff;

      // maintain the taskQ here.
      // 新加入的集合通信放在末位，最后执行。如果新加入的集合通信存在于当前的blkStatus.activeCollIds里边，也不必强行放到末位。
      int new_numActiveColls = 0;
      bool newActiveCollId_in_taskQ = false;
      // TODO: 考虑循环展开的优化。
      for (int i = 0; i < blkStatus.numActiveColls; ++i) {
        if (blkStatus.activeCollIds[i] == newActiveCollId) {
          newActiveCollId_in_taskQ = true;
        }
        CollCtx *globalCollCtx4Blk7OldActiveColl = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + blkStatus.activeCollIds[i];
        if (globalCollCtx4Blk7OldActiveColl->executing == 1) {
          // 在同一个数组上就地操作。new_numActiveColls一定是<=i的，所以不会有问题。
          blkStatus.activeCollIds[new_numActiveColls++] = blkStatus.activeCollIds[i];
        }
      }
      if (!newActiveCollId_in_taskQ) {
        blkStatus.activeCollIds[new_numActiveColls++] = newActiveCollId;
      }
      
      blkStatus.numActiveColls = new_numActiveColls;
    }
  }
}

static __device__ int loadCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId, int turn) {
  int tid = threadIdx.x;
  // int nthreads = blockDim.x;

  // turn = copyToShmemLoop(&sharedCollCtx.comm, &(globalCollCtx4Blk7Coll->comm), tid, nthreads, turn);
  // turn = copyToShmemLoop(&sharedCollCtx.channel, &(globalCollCtx4Blk7Coll->channel), tid, nthreads, turn);
  // // copyToShmemOneShot(&sharedCollCtx.work, &(globalCollCtx4Blk7Coll->work.elems[0]), tid, nthreads); // TODO: 用了这个会报错misaligned，就先loop吧
  // turn = copyToShmemLoop(&(sharedCollCtx.work.elems[0]), &(globalCollCtx4Blk7Coll->work.elems[0]), tid, nthreads, turn);
  // sharedCollCtx.work.elems[0].header.nWarps = globalCollCtx4Blk7Coll->work.elems[0].header.nWarps;
  // // turn = copyToShmemLoop(&sharedCollCtx.work, &(globalCollCtx4Blk7Coll->work), tid, nthreads, turn);
  // __syncthreads(); // 全部线程都执行，可以使用这个同步。

  if (tid == 0) {
    sharedCollCtx.ringPrev = globalCollCtx4Blk7Coll->ringPrev;
    sharedCollCtx.ringNext = globalCollCtx4Blk7Coll->ringNext;
    sharedCollCtx.ringIndex = globalCollCtx4Blk7Coll->ringIndex;
    sharedCollCtx.devPeers = globalCollCtx4Blk7Coll->devPeers;

    sharedCollCtx.rank = globalCollCtx4Blk7Coll->rank;
    sharedCollCtx.nRanks = globalCollCtx4Blk7Coll->nRanks;
    sharedCollCtx.abortFlag = globalCollCtx4Blk7Coll->abortFlag;

    for (int i = 0; i < NCCL_NUM_PROTOCOLS; i++) {
      sharedCollCtx.buffSizes[i] = globalCollCtx4Blk7Coll->buffSizes[i];
    }

    copyNcclWorkElem(sharedCollCtx.workElem, globalCollCtx4Blk7Coll->workElem);

    // // for debug
    // {
    //   struct ncclPeer *recvPeer = &sharedCollCtx.devPeers[sharedCollCtx.ringPrev];
    //   struct ncclPeer *sendPeer = &sharedCollCtx.devPeers[sharedCollCtx.ringNext];
    //   struct ncclConnInfo *recvConn = &recvPeer->recv[0].conn;
    //   uint64_t head = recvConn->step;
    //   struct ncclConnInfo *sendConn = &sendPeer->send[0].conn;
    //   uint64_t tail = sendConn->step;
    //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d load head = %llu, tail = %llu", sharedCollCtx.rank, blockIdx.x, tid, collId, head, tail);
    // }

    // 加载algo、proto、func相关的运行上下文。
    // TODO: 目前只有simple ring allreduce，之后考虑通用性和扩展性。
    sharedCollCtx.saveCtx7Quit = 0; // 每次加载的时候，重置。
    sharedCollCtx.loadAgain = globalCollCtx4Blk7Coll->loadAgain;
    sharedCollCtx.slice4SimpleGenericOp = globalCollCtx4Blk7Coll->slice4SimpleGenericOp;
    sharedCollCtx.offset4SimpleGenericOp = globalCollCtx4Blk7Coll->offset4SimpleGenericOp;

    // sharedCollCtx.totalSteps4RingAllReduce = 2 * sharedCollCtx.nRanks - 1;
    sharedCollCtx.currentStep4RingAllReduce = globalCollCtx4Blk7Coll->currentStep4RingAllReduce;
    sharedCollCtx.gridOffset4RingAllReduce = globalCollCtx4Blk7Coll->gridOffset4RingAllReduce;
    // __threadfence_block();
  }
  // *(blkStatus.barrierCnt + 0 + 6 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  ofcclBarrier(2);
  // *(blkStatus.barrierCnt + 1 + 6 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
 
  return turn;
}

static __device__ void manipulateCQ7ResetDoneColl(int thrdCudaDev, int doneCollId, CQ *cq, CQE *globalCqes, CollCtx *globalCollCtx4Blk7Coll, CollCtx *globalBlk2CollId2CollCtx) {
  // 协调所有blk，发现所有blk都完成，最后一个blk发送CQE
  int old_counter = atomicAdd(&(globalCqes[doneCollId].counter), 1);
  __threadfence(); // cqes在global memory里边，全部block关心。

  // *(blkStatus.collCounters + 0 + doneCollId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;

  OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prepare %lluth CQE for collId = %d", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->cqePrepareCnt), doneCollId);

  if (old_counter + 1 == sharedBlkCount4Coll[doneCollId]) {
    atomicExch(&globalCqes[doneCollId].counter, 0);

    unsigned long long int *cqeWriteCnt = nullptr;
    // TODO: debug
    CollCtx *globalCollCtx4Blk_0_7Coll = globalBlk2CollId2CollCtx + 0 * MAX_LENGTH + doneCollId;
    cqeWriteCnt = &globalCollCtx4Blk_0_7Coll->cqeWriteCnt;

    while (cqWrite(cq, globalCqes + doneCollId, thrdCudaDev, cqeWriteCnt) == -1) {
    }
    // *(blkStatus.collCounters + 1 + doneCollId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;
    __threadfence();
  }

  // 这里不再给blkStatus.numActiveColls减1，只给executing置0。
  blkStatus.currActiveCollId = -1;

  globalCollCtx4Blk7Coll->executing = 0;
  globalCollCtx4Blk7Coll->loadAgain = 0;
  globalCollCtx4Blk7Coll->saveCtx7Quit = 0;

  // ResetDoneColl
  globalCollCtx4Blk7Coll->slice4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->offset4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->currentStep4RingAllReduce = 0;
  globalCollCtx4Blk7Coll->gridOffset4RingAllReduce = 0;

  // for debug
  // {
  //   struct ncclPeer *recvPeer = &sharedCollCtx.devPeers[sharedCollCtx.ringPrev];
  //   struct ncclPeer *sendPeer = &sharedCollCtx.devPeers[sharedCollCtx.ringNext];
  //   struct ncclConnInfo *recvConn = &recvPeer->recv[0].conn;
  //   uint64_t head = recvConn->step;
  //   struct ncclConnInfo *sendConn = &sendPeer->send[0].conn;
  //   uint64_t tail = sendConn->step;
  //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d done head = %llu, tail = %llu", sharedCollCtx.rank, blockIdx.x, tid, doneCollId, head, tail);
  // }
}

static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int thrdLimit, int collId) {
  // globalCollCtx4Blk7Coll->saveCtx7Quit = sharedCollCtx.saveCtx7Quit;
  globalCollCtx4Blk7Coll->loadAgain = sharedCollCtx.loadAgain;
  globalCollCtx4Blk7Coll->slice4SimpleGenericOp = sharedCollCtx.slice4SimpleGenericOp;
  globalCollCtx4Blk7Coll->offset4SimpleGenericOp = sharedCollCtx.offset4SimpleGenericOp;

  globalCollCtx4Blk7Coll->currentStep4RingAllReduce = sharedCollCtx.currentStep4RingAllReduce;
  globalCollCtx4Blk7Coll->gridOffset4RingAllReduce = sharedCollCtx.gridOffset4RingAllReduce;

  blkStatus.totalCtxSwitchCnt++;
  blkStatus.currActiveCollId = -1;
  
  // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, blkStatus.totalCtxSwitchCnt = %llu, blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, blkStatus.totalCtxSwitchCnt, blkStatus.numActiveColls);
  
  // for debug
  // {
  //   struct ncclPeer *recvPeer = &sharedCollCtx.devPeers[sharedCollCtx.ringPrev];
  //   struct ncclPeer *sendPeer = &sharedCollCtx.devPeers[sharedCollCtx.ringNext];
  //   struct ncclConnInfo *recvConn = &recvPeer->recv[0].conn;
  //   uint64_t head = recvConn->step;
  //   struct ncclConnInfo *sendConn = &sendPeer->send[0].conn;
  //   uint64_t tail = sendConn->step;
  //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d save head = %llu, tail = %llu", sharedCollCtx.rank, blockIdx.x, tid, collId, head, tail);
  // }
}

static __device__ int traverseTaskQ(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collCount, CQ *cq, CQE *globalCqes, int turn) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  *(blkStatus.barrierCnt + 0 + 11 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  if (blkStatus.numActiveColls == 0) {
    *(blkStatus.barrierCnt + 1 + 11 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    return turn;
  }

  // TODO: 循环展开的优化？
  int i = 0;
  for (; i < blkStatus.numActiveColls; i++) {

    // 下边这三个量是不变的。
    int collId = blkStatus.activeCollIds[i];
    int blkLimit = sharedBlkCount4Coll[collId];
    int thrdLimit = sharedThrdCount4Coll[collId];

    *(blkStatus.barrierCnt + 0 + 10 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    *(blkStatus.barrierCnt + 2 + 10 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = collId;

    if (bid < blkLimit) { // blk天然分化，保留这个条件 // TODO: 如果节省if判断对性能有提升，可以改变处理方法，让所有block处理所有的集合通信。不过好像也省不了。。。总得判断。
      // block内全部线程都执行：
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;

      *(blkStatus.barrierCnt + 0 + 17 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      ofcclBarrier(9); // TODO: 这个应该是没必要的。尝试对抗时序问题，在读executing之前，加一个同步
      *(blkStatus.barrierCnt + 3 + 10 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = globalCollCtx4Blk7Coll->executing;
      *(blkStatus.barrierCnt + 1 + 17 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

      if (globalCollCtx4Blk7Coll->executing == 1) {
        // if (tid == 0) { // TODO: 主要是打log用的，不打log可以删掉，省一个if。
          blkStatus.currActiveCollId = collId; // 0号线程修改shmem，应该不用原子操作。
        // }

        // ***** 先准备好sharedCollCtx，全部线程都参与 *****
        // 这个load事实上也只应该影响工作的warp，不过由于是操作shmem，所以其他warp没办法，也会受影响。
        turn = loadCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, collId, turn); // 只load一个到shmem
        
        *(blkStatus.barrierCnt + 0 + 15 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

        // ***** 然后调用ofcclFunc *****
        int wid = threadIdx.x / WARP_SIZE;
        if (wid < sharedCollCtx.workElem.header.nWarps) {
          ofcclFuncs[sharedCollCtx.workElem.header.funcIndex](); // 这里边的调用里不涉及__syncthreads().
        }
        
        *(blkStatus.barrierCnt + 1 + 15 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

        *(blkStatus.barrierCnt + 0 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        ofcclBarrier(3);
        *(blkStatus.barrierCnt + 1 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        if (tid == 0) {
          // 根据sharedCollCtx.saveCtx7Quit的情况进行不同处理。
          // 以下的if-else事实上是和当前的工作的warp相关的，不过选择全部线程同步。
          if (sharedCollCtx.saveCtx7Quit == 1) {
            saveExcutingCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, thrdLimit, collId);
          } else {
            // 把对CQ的操作当做循环任务列表的附加动作吧，完成一个集合通信，就操作相应的CQE。
            // 完成的时候才进行下边的调用，只是保存上下文退出不应该调用。
            manipulateCQ7ResetDoneColl(thrdCudaDev, collId, cq, globalCqes, globalCollCtx4Blk7Coll, globalBlk2CollId2CollCtx);
            // 对于完成执行的集合通信应该不用把shmem里的collCtx写回到global mem里边，sendbuff/recvbuff等下次的SQE传过来，剩下的其他都是些静态配置项。
          }
        }
        *(blkStatus.barrierCnt + 0 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        ofcclBarrier(4);
        *(blkStatus.barrierCnt + 1 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      }
    }

    // *(blkStatus.barrierCnt + 1 + 10 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  }
  *(blkStatus.barrierCnt + 2 + 11 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

  return turn;
}

// TODO: 考虑在按需启停的场景下，会多次启动，执行上会不会有什么变化。
__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *globalVolunteerQuit, int *finallyQuit, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    blkStatus.quit = 0;
    
#ifdef ARRAY_DEBUG_ON
    blkStatus.barrierCnt = barrierCnt;
    blkStatus.collCounters = collCounters;
#endif
    BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
    if (myGlobalBlkStatus->hasVolunteerQuitted == 0) {
      blkStatus.numActiveColls = 0;
      blkStatus.currActiveCollId = -1;
      blkStatus.sqReadFrontier = 0;
      blkStatus.hasVolunteerQuitted = 0;

      blkStatus.totalCtxSwitchCnt = 0;
      blkStatus.totalVolunteerQuitCnt = 0;
    } else { // 从volunteer quit恢复回来
      blkStatus.numActiveColls = myGlobalBlkStatus->numActiveColls;
      for (int i = 0; i < blkStatus.numActiveColls; ++i) {
        blkStatus.activeCollIds[i] = myGlobalBlkStatus->activeCollIds[i];
      }
      blkStatus.currActiveCollId = myGlobalBlkStatus->currActiveCollId;
      blkStatus.sqReadFrontier = myGlobalBlkStatus->sqReadFrontier;
      blkStatus.hasVolunteerQuitted = 1;

      blkStatus.totalCtxSwitchCnt = myGlobalBlkStatus->totalCtxSwitchCnt;
      blkStatus.totalVolunteerQuitCnt = myGlobalBlkStatus->totalVolunteerQuitCnt;
    }

    // bugfix: 原来这里写的是globalVolunteerQuit = 0。。。。。。
    *globalVolunteerQuit = 0; // 在tid == 0里做了，出去还有同步，就不限制
  }
  ofcclBarrier(5);
  *(blkStatus.barrierCnt + 0 + 5 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  
  OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel starts, blkStatus.totalVolunteerQuitCnt = %llu, blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, blkStatus.totalVolunteerQuitCnt, blkStatus.numActiveColls);
  // __syncwarp(); // ！！！！！！为了打印log加的！！！！
  
  // int tempRound = 0;
  int turn = 0;

  turn = initContexts(thrdCudaDev, collCount, globalBlkCount4Coll, globalThrdCount4Coll, globalCollIds, globalDevComm7WorkElems, globalBlk2CollId2CollCtx, turn);
  
  int checkSQ7TidyTaskQFailCnt = 0;
  while (true) {
    for (int i = 0; i < TRAVERSE_TIMES; i++) {
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before traverseTaskQ, (%d / %d), blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, i, TRAVERSE_TIMES, blkStatus.numActiveColls);
      // __syncwarp(); // ！！！！！！为了打印log加的！！！！

      turn = traverseTaskQ(thrdCudaDev, globalBlk2CollId2CollCtx, collCount, cq, globalCqes, turn);
      
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, traverseTaskQ return, (%d / %d)", thrdCudaDev, blockIdx.x, tid, i, TRAVERSE_TIMES);
      // __syncwarp(); // ！！！！！！为了打印log加的！！！！  
    }
    
    // OFCCL_LOG_WARP_HEAD(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> before checkSQ7TidyTaskQ, blkStatus.numActiveColls = %d, totalVolunteerQuitCnt = %llu", thrdCudaDev, bid, threadIdx.x, blkStatus.numActiveColls, blkStatus.totalVolunteerQuitCnt);

    *(blkStatus.barrierCnt + 0 + 12 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    ofcclBarrier(6);
    *(blkStatus.barrierCnt + 1 + 12 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

    if (tid == 0) {
      OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before checkSQ7TidyTaskQ, sqReadFrontier = %llu, sq->head=%llu, sq->tail=%llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, blkStatus.sqReadFrontier), DevLogicSqHeadInline(sq), DevLogicSqTailInline(sq));
      checkSQ7TidyTaskQ(thrdCudaDev, sq, globalBlk2CollId2CollCtx, &checkSQ7TidyTaskQFailCnt, finallyQuit);
      
      // 只有0号线程才会执行checkSQ7TidyTaskQ，自然只有0号线程才会更改checkSQ7TidyTaskQFailCnt，并且进行相应调整。

      // checkSQ7TidyTaskQFailCnt = 0; // TODO: 禁止主动退出；本来想用ParseBooleanFromEnv这样的方法用env控制，不过是device函数，还是算了。

      if (checkSQ7TidyTaskQFailCnt > TOLERANT_FAIL_CHECK_SQ_CNT) {
        // 主动退出。
        if (bid == 0) { // 区别对待0号blk和其他。0号决定退出，其他才可以退。
          atomicExch(globalVolunteerQuit, 1); // 这个原子操作还是有必要的，要给其他block看。
        }
        if (*globalVolunteerQuit == 1) { // 0和其他blk都进入这里。编号较大的blk在0号block没退出的情况下，可以继续循环执行checkSQ7TidyTaskQ，可以发现blkStatus.sqReadFrontier < sq->head，从而将checkSQ7TidyTaskQFailCnt置零。
          BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;

          myGlobalBlkStatus->hasVolunteerQuitted = 1;
          blkStatus.quit = 1;
          ++blkStatus.totalVolunteerQuitCnt;

          // 保存blkstatus
          myGlobalBlkStatus->numActiveColls = blkStatus.numActiveColls;
          for (int i = 0; i < blkStatus.numActiveColls; ++i) {
            myGlobalBlkStatus->activeCollIds[i] = blkStatus.activeCollIds[i];
          }
          myGlobalBlkStatus->currActiveCollId = blkStatus.currActiveCollId;
          myGlobalBlkStatus->sqReadFrontier = blkStatus.sqReadFrontier;
          myGlobalBlkStatus->totalCtxSwitchCnt = blkStatus.totalCtxSwitchCnt;
          myGlobalBlkStatus->totalVolunteerQuitCnt = blkStatus.totalVolunteerQuitCnt;

          OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, Volunteer Quit, checkSQ7TidyTaskQFailCnt = %d, blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, checkSQ7TidyTaskQFailCnt, blkStatus.numActiveColls);
        }
      }
    }

    *(blkStatus.barrierCnt + 0 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    ofcclBarrier(7); // prims_simple里用的是8和15。
    *(blkStatus.barrierCnt + 1 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    
    // daemonKernel一开始这个数组用不上，可以用来记点其他信息
    *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 33 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.totalCtxSwitchCnt;
    *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 34 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.totalVolunteerQuitCnt;
    *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 35 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.numActiveColls;

    // 记录数组的前10项，未必都是有效的。所有线程都做，看到的应该是一样的。
    for (int i = 0; i < PrintTestQNum; i++) {
      *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + (36 + i) * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.activeCollIds[i];
    }


    
    if (blkStatus.quit == 1) {
      if (*finallyQuit == 1) {
        OFCCL_LOG_THRD_0(OFCCL_FINAL_OR_VOLUNTEER_QUIT, "Rank<%d> Blk<%d> Thrd<%d> collCount=%d, totalCtxSwitchCnt=%llu, totalVolunteerQuitCnt=%llu", thrdCudaDev, bid, tid, collCount, blkStatus.totalCtxSwitchCnt, blkStatus.totalVolunteerQuitCnt);
        // OFCCL_LOG_BLK_0_THRD_0(OFCCL_FINAL_OR_VOLUNTEER_QUIT, "Rank<%d> Blk<%d> Thrd<%d> collCount=%d, totalCtxSwitchCnt=%llu", thrdCudaDev, bid, tid, collCount, blkStatus.totalCtxSwitchCnt);
      }
      
      *(blkStatus.barrierCnt + 1 + 5 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      return;
    }
  }
}