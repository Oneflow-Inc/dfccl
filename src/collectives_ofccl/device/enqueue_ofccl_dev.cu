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
//     // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, we->redOpArgIsPtr = %d, we->redOpArg = %llu", sharedCollCtx.comm.rank, blockIdx.x, threadIdx.x, we->redOpArgIsPtr, we->redOpArg);
//   }
// }

// share mem用超了。
// TODO: 可以不同的algo、proto使用不同的数据类型，不过可以看看是不是有意义
__shared__ CollCtx sharedCollCtx; // 不能static，primitives要用

static __shared__ BlkStatus blkStatus;
// TODO: 下边这几个可以尝试用constant，先不急
static __shared__ int sharedCollIds[MAX_LENGTH]; // prepareColl会接受用户传进来的collId，而prepareColl工作在每个rank上，我们不能假设各个rank会收到连续的collId，所以用一个数组把收到的collId整理起来，其实相当于是维护了一个map，但是cuda上没有map，只好用这种方式
static __shared__ int sharedBlkCount4Coll[MAX_LENGTH];
static __shared__ int sharedThrdCount4Coll[MAX_LENGTH];

static __device__ int sqRead(SQ *sq, unsigned long long int sqReadFrontier, SQE *target, int thrdCudaDev) {
  int bid = blockIdx.x;
  int sqeCollId;
  
  // int tid = threadIdx.x;
  // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> enter sqRead, sqHead=%llu, sqTail=%llu, empty=%d, RingBuffer_get(sq, sqReadFrontier)->counter=%d, RingBuffer_get(sq, sqReadFrontier)->collId=%d, RingBuffer_get(sq, sqReadFrontier)->quit=%d, RingBuffer_get(sq, sqReadFrontier)->logicHead=%d, GetLogicFrontier(sq, sqReadFrontier)=%llu", thrdCudaDev, bid, tid, RingBuffer_logic_head(sq), RingBuffer_logic_tail(sq), RingBuffer_empty(sq), RingBuffer_get(sq, sqReadFrontier)->counter, RingBuffer_get(sq, sqReadFrontier)->collId, RingBuffer_get(sq, sqReadFrontier)->quit, RingBuffer_get(sq, sqReadFrontier)->logicHead, GetLogicFrontier(sq, sqReadFrontier));
  if (RingBuffer_empty(sq)) {
    return -1;
  }
  // 先读过来，然后再判断，最后更新状态：sqe->counter; 以及在恰当的时候commit read
  *target = *RingBuffer_get(sq, sqReadFrontier);
  if (target->quit) {
    // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Get quit", thrdCudaDev, bid, tid);
    return 0;
  }

  // 先判断一下相应的collId是不是该自己的bid处理，不该自己处理直接返回-1
  sqeCollId = target->collId;
  // OFCCL_LOG(OFCCL, "Blk<%d>, Thrd<%d> sharedBlkCount4Coll[%d]=%d", thrdCudaDev, bid, tid, sqeCollId, sharedBlkCount4Coll[sqeCollId]);
  if (bid >= sharedBlkCount4Coll[sqeCollId]) {
    return -1;
  } else {
    // 自己读到之后，更新相应的counter；至于读到的sqe对应的collId是不是该自己处理，是caller的事。
    // 如果发现自己读完之后，所有block都读了，那么commit read
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> PREPARE to increase counter(curr=%d) for sqe of collId %d", thrdCudaDev, bid, tid, RingBuffer_get(sq, sqReadFrontier)->counter, sqeCollId);
    int old_counter = atomicAdd(&(RingBuffer_get(sq, sqReadFrontier)->counter), 1);
    __threadfence_system();
    // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> increase counter to %d for sqe of collId %d", thrdCudaDev, bid, tid, old_counter + 1, sqeCollId);
    
    if (old_counter + 1 == sharedBlkCount4Coll[sqeCollId]) {
      
      unsigned long long int old_head = atomicAdd(&sq->head, 1);

      __threadfence_system();
      // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> sqe of collId %d commit read, new sqHead is %llu", thrdCudaDev, bid, tid, sqeCollId, old_head + 1);
    }
  }
  
  return 0;
}

static __device__ int cqWrite(CQ *cq, CQE *cqe, int thrdCudaDev) {
  if (RingBuffer_full(cq)) {
    // not an error; caller keeps trying.
    return -1;
  }

  *RingBuffer_get_tail(cq) = *cqe;

  __threadfence_system();

  atomicAdd(&cq->tail, 1); // uint64, 一往无前++
  // RingBuffer_commit_write(cq, 1);

  OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> cqWrite done, RingBuffer_full(cq)=%d, cqHead=%llu, cqTail=%llu", thrdCudaDev, blockIdx.x, threadIdx.x, RingBuffer_full(cq), RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));

  return 0;
}


static __device__ int initContexts(int thrdCudaDev, int collCount, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int turn, int *volunteerQuit) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  // 构建任务列表
  for (int i = 0; i < collCount; i++) {
    int collId = sharedCollIds[i] = globalCollIds[i];
    // 以下这两个变量会限制很多行为。
    int blkLimit = sharedBlkCount4Coll[collId] = globalBlkCount4Coll[collId];
    sharedThrdCount4Coll[collId] = globalThrdCount4Coll[collId];

    // 下边这部分逻辑在在volunteerQuit == 1的情况下不执行。
    if (*volunteerQuit == 1) {
      // 每个block一份globalShmem
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;

      // ***** 移植ncclKernel的逻辑 *****
      if (bid < blkLimit) {
        ncclDevComm *comm = globalDevComm7WorkElems[collId].comm;
        turn = copyToShmemLoop(&(globalCollCtx4Blk7Coll->comm), comm, tid, nthreads, turn);
        // 一个奇技淫巧：get address of channel without incurring indirect load from ncclDevComm::channels
        // 这里通过bid选择了合适的channel，很多集合通信真正执行时用到的硬件信息就存在channel里边。
        ncclChannel *channel = &((ncclDevCommAndChannels*)comm)->channels[bid];
        turn = copyToShmemLoop(&(globalCollCtx4Blk7Coll->channel), channel, tid, nthreads, turn); // 尝试使用oneshot，会报错warp misaligned，所以看来必须用loop。

        // nccl中限制只在bid=0里进行这样的拷贝，对于ofccl而言，ofcclShmem就是任务列表，所以对于所有的线程，我们都把同样的work存进去；
        turn = copyToShmemLoop(&(globalCollCtx4Blk7Coll->work.elems[0]), &(globalDevComm7WorkElems[collId].first), tid, nthreads, turn); // nccl 2.12里边这地方用copyToShmemOneShot进行拷贝，但是oneShot的实现使用了与shared mem相关的内联汇编，所以这里也使用loop进行拷贝。
        // nccl中接下来要处理channel.workFifoDev，然而对于目前的ofccl，只处理first就好，channel.workFifoDev不会有其他任务了。
        __syncthreads();

        // if (globalCollCtx4Blk7Coll->work.header.type == ncclWorkTypeColl) {
        //   // #define NCCL_MAX_WORK_ELEMENTS (NCCL_WORK_SIZE / sizeof(struct ncclWorkElem))=512/64=8
        //   // 原来这个写法，应该是想修改we->redOpArg，不过修改we->redOpArg一个线程就够了，所以让理论上最多的线程来工作，咱们保留就好。
        //   if (tid < NCCL_MAX_WORK_ELEMENTS) ofcclRedopPtrDeref(&(globalCollCtx4Blk7Coll->work.elems[tid]));
        // } // 目前不用考虑其他ncclWorkType
        // __syncthreads();
        
        if (tid == 0) {
          globalCollCtx4Blk7Coll->executing = 0;
          // globalCollCtx4Blk7Coll->numDoneThrds = 0;
          
          globalBlk2CollId2CollCtx->saveCtx7Quit = 0;
          globalBlk2CollId2CollCtx->loadAgain = 0;
          globalBlk2CollId2CollCtx->slice4SimpleGenericOp = 0;
          globalBlk2CollId2CollCtx->offset4SimpleGenericOp = 0;

          globalBlk2CollId2CollCtx->currentStep4RingAllReduce = 0;
          globalBlk2CollId2CollCtx->gridOffset4RingAllReduce = 0;

          // OFCCL_LOG(OFCCL, "nthreads: globalCollCtx4Blk7Coll->work.elems[0].nWarps*WARP_SIZE=%d, thrdLimit=%d", globalCollCtx4Blk7Coll->work.elems[0].header.nWarps*WARP_SIZE, thrdLimit);
          
          // 整个grid里出一个线程来设置就好了。
          if (bid == 0) {
            *volunteerQuit = 0;
          }
        }
        __syncthreads();
      }
    }
  }
  return turn;
}

// 为了初步实现按需启停，增加一个“空read计数，读不到新的，增加计数”
static __device__ void checkSQ(int thrdCudaDev, SQ *sq, CollCtx *globalBlk2CollId2CollCtx, int *failCnt) {
  int bid = blockIdx.x;
  // int tempThrdCudaDev = thrdCudaDev;
  
  SQE target;
  // TODO: really need system?? 之后可以看看__threadfence()会不会提高性能。
  __threadfence_system(); // make sure read new head and tail.

  // OFCCL_LOG_BLK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, sq @ %p", thrdCudaDev, bid, threadIdx.x, sq);
  OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, sq->head = %llu, sq->tail = %llu, blkStatus.numActiveColls = %d", thrdCudaDev, bid, threadIdx.x, sq->head, sq->tail, blkStatus.numActiveColls);

  if (blkStatus.sqReadFrontier < sq->head) {
    // 如果当前bid比较大，一些SQE不需要这个block处理，就会跳过。导致当前block的frontier小于head。
    // 不给sqRead增加返回值种类；否则会增加无谓的sqRead调用、增加访存次数。
    blkStatus.sqReadFrontier = sq->head;
  }

  // 能读到，假如是正常SQE，把信息在任务列表里记录一下；假如是quit，那也记录一下
  // 读不到新东西那就算了
  if (RingBuffer_logic_tail(sq) == GetLogicFrontier(sq, blkStatus.sqReadFrontier) || sqRead(sq, blkStatus.sqReadFrontier, &target, thrdCudaDev) == -1) {
    *failCnt += 1;
    if (blkStatus.numActiveColls > 0) {
      *failCnt = 0;
    }
    return;
  } else {
    *failCnt = 0;
    blkStatus.sqReadFrontier++;
    if (target.quit) {
      blkStatus.quit = 1;
      // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Main Thrd of Blk quit", thrdCudaDev, bid, threadIdx.x);
      return;
    }

    // 正常读到了SQE的话，需要往global的globalBlk2CollId2CollCtx表项里边写入，更新blkStatus.numActiveColls
    int newActiveCollId = target.collId;
    int blkLimit = sharedBlkCount4Coll[newActiveCollId];
    if (bid < blkLimit) {
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + newActiveCollId;
      if (globalCollCtx4Blk7Coll->executing == 1) {
        OFCCL_LOG(OFCCL_FATAL, "Rank<%d> Blk<%d> Thrd<%d> globalCollCtx4Blk7Coll->executing should be 0! sq->head = %llu, sq->tail = %llu, blkStatus.sqReadFrontier = %llu", thrdCudaDev, bid, threadIdx.x, RingBuffer_logic_head(sq), RingBuffer_logic_tail(sq), GetLogicFrontier(sq, blkStatus.sqReadFrontier));
      }
      globalCollCtx4Blk7Coll->executing = 1;
      globalCollCtx4Blk7Coll->work.elems[0].sendbuff = target.sendbuff;
      globalCollCtx4Blk7Coll->work.elems[0].recvbuff = target.recvbuff;
      
      /* IF_CHECK checkSQ里收到sqe的时候浅打印一下初始sendbuff。作用一般。*/

      // float *sendptr = (float *)target.sendbuff;
      // for (int i = 0; i < buffPrintNum; i++) {
      //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> sendbuff @%p sendbuff[%d]=%f", thrdCudaDev, bid, threadIdx.x, target.sendbuff, i, *(sendptr + i));
      // }
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> recvbuff @%p", thrdCudaDev, bid, threadIdx.x, target.recvbuff);

      // block的0号线程操作shmem，不用原子操作
      blkStatus.numActiveColls += 1;
      __threadfence_block();

      OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> get collId %d, blkStatus.sqReadFrontier updates to %llu, blkStatus.numActiveColls = %d", thrdCudaDev, bid, threadIdx.x, target.collId, GetLogicFrontier(sq, blkStatus.sqReadFrontier), blkStatus.numActiveColls);
    }
  }
}

static __device__ void manipulateCQ(int thrdCudaDev, int doneCollId, CQ *cq, CQE *globalCqes) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int blkLimit = sharedBlkCount4Coll[doneCollId];
  int thrdLimit = sharedThrdCount4Coll[doneCollId];
  
  if (bid < blkLimit && tid == 0) {
    // 协调所有blk，发现所有blk都完成，最后一个blk发送CQE
    int old_counter = atomicAdd(&(globalCqes[doneCollId].counter), 1);
    __threadfence(); // cqes在global memory里边，全部thread关心。

    if (old_counter + 1 == sharedBlkCount4Coll[doneCollId]) {
      atomicExch(&globalCqes[doneCollId].counter, 0);
      while (cqWrite(cq, globalCqes + doneCollId, thrdCudaDev) == -1) {
        // tempRound++;
        // if(tempRound % tempPrintRound == 0) {
        //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> cqWrite fail, RingBuffer_full(cq)=%d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, RingBuffer_full(cq), RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
        // }

      }
      OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> insert CQE for collId %d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, doneCollId, RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
      
      __threadfence();
    }
  }
  ofcclBarrier(OFCCL_SYNC_COLL_WORKER_BAR_ID, thrdLimit);
}

static __device__ int loadCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId, int turn) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  turn = copyToShmemLoop(&sharedCollCtx.comm, &(globalCollCtx4Blk7Coll->comm), tid, nthreads, turn);
  turn = copyToShmemLoop(&sharedCollCtx.channel, &(globalCollCtx4Blk7Coll->channel), tid, nthreads, turn);
  // copyToShmemOneShot(&sharedCollCtx.work, &(globalCollCtx4Blk7Coll->work.elems[0]), tid, nthreads); // TODO: 用了这个会报错misaligned，就先loop吧
  turn = copyToShmemLoop(&sharedCollCtx.work.elems[0], &(globalCollCtx4Blk7Coll->work.elems[0]), tid, nthreads, turn);
  __syncthreads(); // 全部线程都执行，可以使用这个同步。
  
  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before ofcclRedopPtrDeref, NCCL_MAX_WORK_ELEMENTS = %lu", thrdCudaDev, blockIdx.x, tid, NCCL_MAX_WORK_ELEMENTS);
  // __syncthreads(); // ！！！！！！为了打印log加的！！！！
  
  // if (sharedCollCtx.work.header.type == ncclWorkTypeColl) {
  //   if (tid < NCCL_MAX_WORK_ELEMENTS) ofcclRedopPtrDeref(&(sharedCollCtx.work.elems[tid]));
  // } // TODO: 目前不用考虑其他ncclWorkType
  // __syncthreads(); // initContexts 在调用ofcclRedopPtrDeref加上了同步。本来也该加。
  
  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, after ofcclRedopPtrDeref", thrdCudaDev, blockIdx.x, tid);
  // __syncthreads(); // ！！！！！！为了打印log加的！！！！

  if (tid == 0) {
    // TODO: 目前只有simple ring allreduce，之后考虑通用性和扩展性。
    // 加载algo、proto、func相关的运行上下文。

    // sharedCollCtx.saveCtx7Quit = globalCollCtx4Blk7Coll->saveCtx7Quit; // 这个看起来也可以充当标记是否是跑了一半的标记位
    sharedCollCtx.saveCtx7Quit = 0; // 每次加载的时候，重置。
    sharedCollCtx.loadAgain = globalCollCtx4Blk7Coll->loadAgain;
    sharedCollCtx.slice4SimpleGenericOp = globalCollCtx4Blk7Coll->slice4SimpleGenericOp;
    sharedCollCtx.offset4SimpleGenericOp = globalCollCtx4Blk7Coll->offset4SimpleGenericOp;

    // sharedCollCtx.totalSteps4RingAllReduce = 2 * sharedCollCtx.comm.nRanks - 1;
    sharedCollCtx.currentStep4RingAllReduce = globalCollCtx4Blk7Coll->currentStep4RingAllReduce;
    sharedCollCtx.gridOffset4RingAllReduce = globalCollCtx4Blk7Coll->gridOffset4RingAllReduce;
  }
  __syncthreads();
  
  return turn;
}

static __device__ void resetDoneColl(int thrdCudaDev, int doneCollId, CollCtx *globalCollCtx4Blk7Coll) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int blkLimit = sharedBlkCount4Coll[doneCollId];
  int thrdLimit = sharedThrdCount4Coll[doneCollId];

  if (bid < blkLimit && tid == 0) {
    blkStatus.numActiveColls -= 1;
    blkStatus.currActiveCollId = -1;

    globalCollCtx4Blk7Coll->executing = 0;
    globalCollCtx4Blk7Coll->loadAgain = 0;
    globalCollCtx4Blk7Coll->saveCtx7Quit = 0;

    // 需要把上下文也重置了，不然多次运行会有问题。
    globalCollCtx4Blk7Coll->slice4SimpleGenericOp = 0;
    globalCollCtx4Blk7Coll->offset4SimpleGenericOp = 0;
    globalCollCtx4Blk7Coll->currentStep4RingAllReduce = 0;
    globalCollCtx4Blk7Coll->gridOffset4RingAllReduce = 0;

    /* IF_CHECK 如果要检查对错，把下边露出来 */

    // float *sendptr = (float *)sharedCollCtx.work.elems[0].sendbuff;
    // float *ptr = (float *)sharedCollCtx.work.elems[0].recvbuff;
    // // for (int i = 0; i < 0+buffPrintNum; i++) {
    // //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> sendbuff @ %p sendbuff[%d]=%f", thrdCudaDev, bid, tid, sharedCollCtx.work.elems[0].sendbuff, i, *(sendptr + i));
    // //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> recvbuff @ %p recvbuff[%d]=%f", thrdCudaDev, bid, tid, sharedCollCtx.work.elems[0].recvbuff, i, *(ptr + i));
    // // }
    // for (int i = buffPrintStart; i < buffPrintStart+buffPrintNum; i++) {
    //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> sendbuff @ %p sendbuff[%d]=%f", thrdCudaDev, bid, tid, sharedCollCtx.work.elems[0].sendbuff, i, *(sendptr + i));
    //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> recvbuff @ %p recvbuff[%d]=%f", thrdCudaDev, bid, tid, sharedCollCtx.work.elems[0].recvbuff, i, *(ptr + i));
    // }

  }
  ofcclBarrier(OFCCL_SYNC_COLL_WORKER_BAR_ID, thrdLimit);
}

static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int thrdLimit) {
  int tid = threadIdx.x;
  if (tid == 0) {
    // globalCollCtx4Blk7Coll->saveCtx7Quit = sharedCollCtx.saveCtx7Quit;
    globalCollCtx4Blk7Coll->loadAgain = sharedCollCtx.loadAgain;
    globalCollCtx4Blk7Coll->slice4SimpleGenericOp = sharedCollCtx.slice4SimpleGenericOp;
    globalCollCtx4Blk7Coll->offset4SimpleGenericOp = sharedCollCtx.offset4SimpleGenericOp;
  
    globalCollCtx4Blk7Coll->currentStep4RingAllReduce = sharedCollCtx.currentStep4RingAllReduce;
    globalCollCtx4Blk7Coll->gridOffset4RingAllReduce = sharedCollCtx.gridOffset4RingAllReduce;

    blkStatus.totalCtxSwitchCnt++;
    blkStatus.currActiveCollId = -1;
    
    // int bid = blockIdx.x;
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, blkStatus.totalCtxSwitchCnt = %llu, blkStatus.numActiveColls = %d", thrdCudaDev, bid, tid, blkStatus.totalCtxSwitchCnt, blkStatus.numActiveColls);
  }
  ofcclBarrier(OFCCL_SYNC_COLL_WORKER_BAR_ID, thrdLimit);
}

static __device__ int traverseGlobalCollCtx(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collCount, CQ *cq, CQE *globalCqes, int turn) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int numSeenActiveColls = 0; // 用这个和blkStatus.numActiveColls 配合，减少下边的循环次数，在block天然分化的前提下，看起来可以直接用。

  __threadfence_block();
  if (blkStatus.numActiveColls == 0) {
    return turn;
  }

  for (int i = 0; i < collCount; i++) {
    if (numSeenActiveColls >= blkStatus.numActiveColls) {
      break;
    }

    // 下边这三个量是不变的。
    int collId = sharedCollIds[i];
    int blkLimit = sharedBlkCount4Coll[collId];
    int thrdLimit = sharedThrdCount4Coll[collId];

    if (bid < blkLimit) { // blk天然分化，保留这个条件
      // block内全部线程都执行：
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;
      if (globalCollCtx4Blk7Coll->executing == 1) {
        if (tid == 0) {
          blkStatus.currActiveCollId = collId; // 0号线程修改shmem，应该不用原子操作。
          // __threadfence_block(); // 主要这里有个fence操作，所以可能会带来分化的危险。
        }
        // __syncthreads(); // 没必要为了这么个玩意浪费同步。这里同步线程，那上边就不用插入fence了。

        numSeenActiveColls++;
        
        // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before loadCollCtx blkStatus.numActiveColls = %d, numSeenActiveColls = %d, thrdLimit = %d, blockDim.x = %d", thrdCudaDev, bid, tid, blkStatus.numActiveColls, numSeenActiveColls, thrdLimit, blockDim.x);
        // __syncthreads(); // ！！！！！！为了打印log加的！！！！

        // ***** 先准备好sharedCollCtx，全部线程都参与 *****
        turn = loadCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, collId, turn); // 只load一个到shmem

        // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, after loadCollCtx", thrdCudaDev, bid, tid);
        // __syncthreads(); // ！！！！！！为了打印log加的！！！！
        
        // 只有真正的工作线程才执行
        if (tid < thrdLimit) {
          // ***** 然后调用ofcclFunc *****

          OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before ofcclFuncs[%d], sharedCollCtx.saveCtx7Quit = %d", thrdCudaDev, bid, tid, sharedCollCtx.work.header.funcIndex, sharedCollCtx.saveCtx7Quit);
          ofcclBarrier(OFCCL_SYNC_COLL_WORKER_BAR_ID, thrdLimit); // ！！！！！！为了打印log加的！！！！

          ofcclFuncs[sharedCollCtx.work.header.funcIndex](); // 这里边的调用里不涉及__syncthreads().
          // 根据sharedCollCtx.saveCtx7Quit的情况进行不同处理。
          // OFCCL_LOG_BLK_0_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, ofcclFuncs[%d]() return", thrdCudaDev, blockIdx.x, threadIdx.x, sharedCollCtx.work.header.funcIndex);
  
          OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, ofcclFuncs returns, before ofcclBarrier, sharedCollCtx.saveCtx7Quit = %d", thrdCudaDev, bid, tid, sharedCollCtx.saveCtx7Quit);
  
          ofcclBarrier(OFCCL_SYNC_COLL_WORKER_BAR_ID, thrdLimit);

          // OFCCL_LOG_BLK_0_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, ofcclBarrier returns", thrdCudaDev, bid, tid);

          if (sharedCollCtx.saveCtx7Quit == 1) {
            saveExcutingCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, thrdLimit);
          } else {
            
            OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, complete collId = %d", thrdCudaDev, bid, tid, collId);
            ofcclBarrier(OFCCL_SYNC_COLL_WORKER_BAR_ID, thrdLimit); // ！！！！！！为了打印log加的！！！！

            // atomicAdd(&sharedCollCtx.numDoneThrds, 1); // 有了线程同步，感觉这个变量在跑到底的时候没啥用。
            // 把对CQ的操作当做循环任务列表的附加动作吧，完成一个集合通信，就操作相应的CQE。
            // 完成的时候才进行下边的调用，只是保存上下文退出不应该调用。
            manipulateCQ(thrdCudaDev, collId, cq, globalCqes);
            resetDoneColl(thrdCudaDev, collId, globalCollCtx4Blk7Coll);
            // 对于完成执行的集合通信应该不用把shmem里的collCtx写回到global mem里边，sendbuff/recvbuff等下次的SQE传过来，剩下的其他都是些静态配置项。
            // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> collId %d done", thrdCudaDev, bid, tid, collId);
          }
        }
  
        __syncthreads(); // thrdLimit内外的所有线程，都到这里同步。

      }
    }
  }

  return turn;
}

// TODO: 考虑在按需启停的场景下，会多次启动，执行上会不会有什么变化。
__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *volunteerQuit) {
  int tid = threadIdx.x;
  SQ *localSq = sq;
  
  OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel starts", thrdCudaDev, blockIdx.x, tid);
  __syncthreads(); // ！！！！！！！为了打印log加的！
  
  // int tempRound = 0;
  int turn = 0;

  turn = initContexts(thrdCudaDev, collCount, globalBlkCount4Coll, globalThrdCount4Coll, globalCollIds, globalDevComm7WorkElems, globalBlk2CollId2CollCtx, turn, volunteerQuit);
  
  if (tid == 0) {
    blkStatus.quit = 0;
    blkStatus.numActiveColls = 0;
    blkStatus.currActiveCollId = -1;
    blkStatus.sqReadFrontier = 0;
    blkStatus.totalCtxSwitchCnt = 0;
    // __threadfence_block();
  }
  __syncthreads();

  int checkSQFailCnt = 0;
  while (true) {
    for (int i = 0; i < TRAVERSE_TIMES; i++) {
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before traverseGlobalCollCtx, (%d / %d), blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, i, TRAVERSE_TIMES, blkStatus.numActiveColls);
      // __syncthreads(); // ！！！！！！！为了打印log加的！

      turn = traverseGlobalCollCtx(thrdCudaDev, globalBlk2CollId2CollCtx, collCount, cq, globalCqes, turn);
      
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, traverseGlobalCollCtx return, (%d / %d)", thrdCudaDev, blockIdx.x, tid, i, TRAVERSE_TIMES);
      // __syncthreads(); // ！！！！！！！为了打印log加的！
    }
    if (tid == 0) {
      // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before checkSQ, sq @ %p, blkStatus.numActiveColls=%d, blkStatus.currActiveCollId=%d, blkStatus.totalCtxSwitchCnt=%d", thrdCudaDev, bid, tid, localSq, blkStatus.numActiveColls, blkStatus.currActiveCollId, blkStatus.totalCtxSwitchCnt);
      
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, before checkSQ, sq @ %p", thrdCudaDev, blockIdx.x, tid, localSq);

      checkSQ(thrdCudaDev, localSq, globalBlk2CollId2CollCtx, &checkSQFailCnt);
      
      // 只有0号线程才会执行checkSQ，自然只有0号线程才会更改checkSQFailCnt，并且进行相应调整。
      if (checkSQFailCnt > TOLERANT_FAIL_CHECK_SQ_CNT) {
        // 主动退出。
        *volunteerQuit = 1; // TODO: 这里很多个block都会设置，需要有一个各个block协同的机制。
        blkStatus.quit = 1;
        OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, volunteerQuit, checkSQFailCnt = %d, blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, checkSQFailCnt, blkStatus.numActiveColls);
      }
    }
    __syncthreads();

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, after checkSQ, blkStatus.sqReadFrontier = %llu, blkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, blkStatus.sqReadFrontier, blkStatus.numActiveColls);
    // __syncthreads(); // ！！！！！！！为了打印log加的！
    
    if (blkStatus.quit == 1) {
      // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> quit", thrdCudaDev, bid, tid);


      OFCCL_LOG_FINAL(OFCCL_FINAL_OR_VOLUNTEER_QUIT, "Rank<%d> Blk<%d> Thrd<%d> collCount=%d, totalCtxSwitchCnt=%llu", thrdCudaDev, blockIdx.x, tid, collCount, blkStatus.totalCtxSwitchCnt);
      return;
    }
  }
}