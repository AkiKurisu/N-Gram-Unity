using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
namespace Kurisu.NGram
{
    /// <summary>
    /// N-Gram resolver for <see cref="NGramJob"/>
    /// </summary>
    public class NGramParallelResolver : INGramResolver, IDisposable
    {
        private NativeArray<int>? history;
        private NativeArray<int>? inference;
        public bool Success { get; private set; }
        public int Result { get; private set; }
        private JobHandle jobHandle;
        private NativeArray<int>? result;
        public int StartGram { get; set; } = 2;
        public int NGram { get; set; } = 8;
        public int Batch { get; set; } = 1;
        public int Length => NGram - StartGram + 1;
        private readonly Dictionary<int, int> resultsMap = new();
        public void Dispose()
        {
            history?.Dispose();
            inference?.Dispose();
        }
        public void Resolve(int[] history, int[] inference)
        {
            Resolve(history, inference, 0, history.Length);
        }
        public void Resolve(int[] history, int[] inference, int historyStartIndex, int historyLength)
        {
            result = new NativeArray<int>(Length, Allocator.TempJob);
            var historyArray = new NativeArray<int>(historyLength, Allocator.TempJob);
            for (int i = 0; i < historyLength; i++)
            {
                historyArray[i] = history[i + historyStartIndex];
            }
            this.history = historyArray;
            this.inference = new NativeArray<int>(inference, Allocator.TempJob);
            jobHandle = new NGramParallelJob()
            {
                History = this.history.Value,
                Inference = this.inference.Value,
                Result = result.Value,
                StartGram = StartGram
            }.Schedule(Length, Batch);
        }
        public void Complete()
        {
            jobHandle.Complete();
            Result = GetMax();
            Success = Result >= 0;
            result.Value.Dispose();
            history.Value.Dispose();
            inference.Value.Dispose();
            inference = null;
            history = null;
            result = null;
        }
        public int GetMax()
        {
            for (int i = 0; i < result.Value.Length; ++i)
            {
                if (result.Value[i] == -1) continue;
                if (resultsMap.ContainsKey(result.Value[i]))
                {
                    resultsMap[result.Value[i]] += 1;
                }
                else
                {
                    resultsMap[result.Value[i]] = 1;
                }
            }
            int maxCount = 0;
            int mostFrequent = 0;
            foreach (var pair in resultsMap)
            {
                if (pair.Value > maxCount)
                {
                    maxCount = pair.Value;
                    mostFrequent = pair.Key;
                }
            }
            return mostFrequent;
        }
    }
}