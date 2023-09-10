using System;
using Unity.Collections;
using Unity.Jobs;
namespace Kurisu.NGram
{
    /// <summary>
    /// An example N-Gram resolver for 8-Gram
    /// </summary>
    public class NGram8Resolver : INGramResolver, IDisposable
    {
        private NativeArray<byte>? history;
        private NativeArray<byte>? inference;
        public bool Success { get; private set; }
        public byte Result { get; private set; }
        private JobHandle jobHandle;
        private NativeArray<int>? result;
        public void Dispose()
        {
            history?.Dispose();
            inference?.Dispose();
        }
        public void Resolve(byte[] history, byte[] inference)
        {
            Resolve(history, inference, 0, history.Length);
        }
        public void Resolve(byte[] history, byte[] inference, int historyStartIndex, int historyLength)
        {
            result?.Dispose();
            result = new NativeArray<int>(1, Allocator.TempJob);
            this.history?.Dispose();
            var historyArray = new NativeArray<byte>(historyLength, Allocator.TempJob);
            for (int i = 0; i < historyLength; i++)
            {
                historyArray[i] = history[i + historyStartIndex];
            }
            this.history = historyArray;
            this.inference?.Dispose();
            this.inference = new NativeArray<byte>(inference, Allocator.TempJob);
            jobHandle = new NGram4Job()
            {
                History = this.history.Value,
                Inference = this.inference.Value,
                Result = result.Value,
                NGram = 8
            }.Schedule();
        }
        public void Complete()
        {
            jobHandle.Complete();
            Success = result.Value[0] >= 0;
            if (Success)
                Result = BitConverter.GetBytes(result.Value[0])[7];
            result.Value.Dispose();
            history.Value.Dispose();
            inference.Value.Dispose();
            inference = null;
            history = null;
            result = null;
        }
    }
}