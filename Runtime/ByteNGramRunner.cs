using System.Collections;
using UnityEngine;
namespace Kurisu.NGram
{
    //Example for predict byte using N-Gram
    public class ByteNGramRunner : MonoBehaviour
    {
        private NGram4Resolver resolver;
        private IEnumerator Start()
        {
            byte[] history = { 0, 1, 2, 3, 1, 5, 6, 2, 3, 4, 3, 2, 3, 4, 9, 10 };
            byte[] inference = { 6, 2, 3 };
            resolver = new NGram4Resolver();
            resolver.Resolve(history, inference);
            yield return new WaitForEndOfFrame();
            resolver.Complete();
            Debug.Log("Predict:" + resolver.Result);
        }
        private void OnDestroy()
        {
            resolver.Dispose();
        }
    }
}