using System.Collections;
using System.Text;
using UnityEngine;
namespace Kurisu.NGram
{
    //Example for predict int using N-Gram
    public class IntNGramRunner : MonoBehaviour
    {
        private INGramResolver resolver;
        [SerializeField, Range(4, 12)]
        private int nGram = 8;
        [SerializeField, Range(10, 100)]
        private int idRange = 100;
        private IEnumerator Start()
        {
            Debug.Log("History:");
            int[] history = RandomID(2000);
            Debug.Log("Inference:");
            int[] inference = RandomID(nGram - 1);
            resolver = new NGramParallelResolver() { NGram = nGram };
            //Use last as inference
            resolver.Resolve(history, inference);
            yield return new WaitForEndOfFrame();
            resolver.Complete();
            Debug.Log("Predict:" + resolver.Result);
        }
        private int[] RandomID(int length)
        {
            int[] id = new int[length];
            StringBuilder stringBuilder = new(length * 2);
            for (int i = 0; i < id.Length; ++i)
            {
                id[i] = Random.Range(0, idRange);
                stringBuilder.Append(id[i]);
                stringBuilder.Append(' ');
            }
            Debug.Log(stringBuilder.ToString());
            return id;
        }
        private void OnDestroy()
        {
            resolver.Dispose();
        }
    }
}