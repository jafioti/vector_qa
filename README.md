# vector_qa
Scalable Neural Question Answering

This project contains a transformer-based neural network trained on the Squad 2.0 dataset. The network is forced to generate question and answer vectors to represent the question and possible answers, which are then compared to find the closest vector. The loss function maximises the distance between the question vector and the answer vectors that do not contain the answer, while minimising the distance between the question vector and the answer vector containing the answer. 

This setup allows the model to leverage large-scale QA datasets, while retaining the scalability of vector-similarity scorers. With a library such as FAISS, this system scales to millions or billions of sources of information for question answering, making open-domain QA possible. Most QA systems don't have this ability, as they need to consume the entire question and potential answers at once, meaning that efficient vector-similarity matching cannot be done.
