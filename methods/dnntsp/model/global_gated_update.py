import torch
import torch.nn as nn


class global_gated_update(nn.Module):

    def __init__(self, items_total, item_embedding):
        super(global_gated_update, self).__init__()
        self.items_total = items_total
        self.item_embedding = item_embedding

        # alpha -> the weight for updating
        self.alpha = nn.Parameter(torch.rand(items_total, 1), requires_grad=True)

    def forward(self, graph, nodes, nodes_output):
        """
        :param graph: batched graphs, with the total number of nodes is `node_num`,
                        including `batch_size` disconnected subgraphs
        :param nodes: tensor (n_1+n_2+..., )
        :param nodes_output: the output of self-attention model in time dimension, (n_1+n_2+..., F)
        :return:
        """

        num_nodes_per_graph = graph.ptr[1:] - graph.ptr[:-1]  # Equivalent to `batch_num_nodes()`
        batch_embedding = []
        start_idx = 0  # Start index for slicing nodes in a batch

        # Get the global item embeddings
        items_embedding = self.item_embedding(torch.arange(self.items_total, device=nodes.device))

        for num_nodes in num_nodes_per_graph:
            # Slice node features for this graph
            output_node_features = nodes_output[start_idx:start_idx + num_nodes, :]  # Shape: (user_nodes, item_embed_dim)
            output_nodes = nodes[start_idx:start_idx + num_nodes]  # Nodes for this subgraph

            # Initialize beta (items_total, 1) and set indicators
            beta = torch.zeros(self.items_total, 1, device=nodes.device)
            beta[output_nodes] = 1

            # Compute updated embedding using gated mechanism
            embed = (1 - beta * self.alpha) * items_embedding.clone()

            # Apply gated update for appearing items
            embed[output_nodes, :] = embed[output_nodes, :] + self.alpha[output_nodes] * output_node_features

            # Append processed embedding for this batch
            batch_embedding.append(embed)

            start_idx += num_nodes  # Move to the next graph in the batch

        # Stack embeddings into shape (batch_size, items_total, item_embed_dim)
        batch_embedding = torch.stack(batch_embedding)
        
        return batch_embedding

        # dgl version
        # nums_nodes, id = graph.batch_num_nodes(), 0
        # items_embedding = self.item_embedding(torch.tensor([i for i in range(self.items_total)]).to(nodes.device))
        # batch_embedding = []
        # for num_nodes in nums_nodes:
        #     # tensor, shape, (user_nodes, item_embed_dim)
        #     output_node_features = nodes_output[id:id + num_nodes, :]
        #     # get each user's nodes
        #     output_nodes = nodes[id: id + num_nodes]
        #     # beta, tensor, (items_total, 1), indicator vector, appear item -> 1, not appear -> 0
        #     beta = torch.zeros(self.items_total, 1).to(nodes.device)
        #     beta[output_nodes] = 1
        #     # update global embedding by gated mechanism
        #     # broadcast (items_total, 1) * (items_total, item_embed_dim) -> (items_total, item_embed_dim)
        #     embed = (1 - beta * self.alpha) * items_embedding.clone()
        #     # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
        #     embed[output_nodes, :] = embed[output_nodes, :] + self.alpha[output_nodes] * output_node_features
        #     batch_embedding.append(embed)
        #     id += num_nodes
        # # (B, items_total, item_embed_dim)
        # batch_embedding = torch.stack(batch_embedding)
        # return batch_embedding
