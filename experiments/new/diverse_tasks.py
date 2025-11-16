"""
Diverse Task Implementations for TAP Experiments
==================================================

Extends TAP experiments beyond vision tasks to:
1. NLP Tasks (text classification, sequence labeling)
2. RL Tasks (policy learning with dimensionality tracking)
3. Vision-Language Tasks (image captioning, VQA)

This enables testing TAP dynamics across different learning paradigms.

Author: Javier Marín
Date: 2024-11-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json

# ==============================================================================
# NLP TASKS
# ==============================================================================

class TextClassificationDataset(Dataset):
    """
    Simple text classification dataset.
    Can be used for sentiment analysis, topic classification, etc.
    """

    def __init__(self, texts: List[str], labels: List[int],
                 vocab: Optional[Dict[str, int]] = None,
                 max_length: int = 128):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels
            vocab: Vocabulary mapping word -> index (None = build from texts)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

        self.encoded_texts = [self._encode_text(text) for text in texts]

    def _build_vocab(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """Build vocabulary from texts."""
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Special tokens
        vocab = {'<PAD>': 0, '<UNK>': 1}

        # Add frequent words
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor of indices."""
        words = text.lower().split()[:self.max_length]
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

        # Pad to max_length
        if len(indices) < self.max_length:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.encoded_texts[idx], self.labels[idx]


class SequenceLabelingDataset(Dataset):
    """
    Dataset for sequence labeling tasks (NER, POS tagging, etc.)
    """

    def __init__(self, sequences: List[List[str]],
                 labels: List[List[int]],
                 vocab: Optional[Dict[str, int]] = None,
                 label_vocab: Optional[Dict[str, int]] = None,
                 max_length: int = 128):
        """
        Args:
            sequences: List of token sequences
            labels: List of label sequences (same length as tokens)
            vocab: Vocabulary for tokens
            label_vocab: Vocabulary for labels
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

        if vocab is None:
            self.vocab = self._build_vocab([word for seq in sequences for word in seq])
        else:
            self.vocab = vocab

        if label_vocab is None:
            self.label_vocab = {label: idx for idx, label in
                               enumerate(sorted(set(l for seq in labels for l in seq)))}
        else:
            self.label_vocab = label_vocab

        self.encoded_data = [self._encode_sequence(seq, labs)
                            for seq, labs in zip(sequences, labels)]

    def _build_vocab(self, words: List[str]) -> Dict[str, int]:
        """Build vocabulary from words."""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        unique_words = sorted(set(words))
        for word in unique_words:
            vocab[word] = len(vocab)
        return vocab

    def _encode_sequence(self, sequence: List[str], labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence and labels to tensors."""
        # Truncate if needed
        sequence = sequence[:self.max_length]
        labels = labels[:self.max_length]

        # Encode tokens
        token_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in sequence]

        # Pad
        seq_len = len(token_ids)
        if seq_len < self.max_length:
            token_ids += [self.vocab['<PAD>']] * (self.max_length - seq_len)
            labels += [0] * (self.max_length - seq_len)  # Pad label

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_data[idx]


class TextClassificationModel(nn.Module):
    """
    Simple text classification model with embeddings + LSTM/Transformer.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 2,
                 num_layers: int = 2, encoder_type: str = 'lstm',
                 dropout: float = 0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for encoder
            num_classes: Number of output classes
            num_layers: Number of encoder layers
            encoder_type: 'lstm' or 'transformer'
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_type = encoder_type

        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                  bidirectional=True)
            classifier_input_dim = hidden_dim * 2  # Bidirectional
        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=4, dim_feedforward=hidden_dim,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            classifier_input_dim = embedding_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] token indices
        Returns:
            [batch_size, num_classes] logits
        """
        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Encode
        if self.encoder_type == 'lstm':
            encoded, (hidden, _) = self.encoder(embedded)
            # Use final hidden states from both directions
            sentence_repr = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, hidden*2]
        else:  # transformer
            encoded = self.encoder(embedded)  # [batch, seq_len, embed_dim]
            # Use mean pooling
            sentence_repr = encoded.mean(dim=1)  # [batch, embed_dim]

        sentence_repr = self.dropout(sentence_repr)
        logits = self.classifier(sentence_repr)

        return logits


class SequenceLabelingModel(nn.Module):
    """
    Sequence labeling model (NER, POS tagging, etc.)
    """

    def __init__(self, vocab_size: int, num_labels: int,
                 embedding_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            num_labels: Number of output labels per token
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] token indices
        Returns:
            [batch_size, seq_len, num_labels] logits for each token
        """
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        encoded, _ = self.lstm(embedded)  # [batch, seq_len, hidden*2]
        encoded = self.dropout(encoded)
        logits = self.classifier(encoded)  # [batch, seq_len, num_labels]

        return logits


# ==============================================================================
# RL TASKS
# ==============================================================================

class PolicyNetwork(nn.Module):
    """
    Simple policy network for discrete action spaces.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu'):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh')
        """
        super().__init__()

        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            [batch_size, action_dim] action logits
        """
        return self.network(state)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: [batch_size, state_dim]
            deterministic: If True, return argmax action

        Returns:
            actions: [batch_size] sampled actions
            log_probs: [batch_size] log probabilities
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            actions = torch.argmax(probs, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values.
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            [batch_size, 1] state values
        """
        return self.network(state)


class RLEnvironmentWrapper:
    """
    Wrapper for RL environments (Gym-like interface).
    Provides simple environments for testing TAP dynamics.
    """

    def __init__(self, env_name: str = 'cartpole'):
        """
        Args:
            env_name: 'cartpole', 'mountaincar', or 'simple_grid'
        """
        self.env_name = env_name

        if env_name == 'cartpole':
            self.state_dim = 4
            self.action_dim = 2
        elif env_name == 'mountaincar':
            self.state_dim = 2
            self.action_dim = 3
        elif env_name == 'simple_grid':
            self.state_dim = 2
            self.action_dim = 4
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        if self.env_name == 'cartpole':
            self.state = np.random.uniform(-0.05, 0.05, size=4)
            self.steps = 0
        elif self.env_name == 'mountaincar':
            self.state = np.array([np.random.uniform(-0.6, -0.4), 0.0])
            self.steps = 0
        elif self.env_name == 'simple_grid':
            self.state = np.array([0, 0])  # Start at origin
            self.goal = np.array([5, 5])
            self.steps = 0

        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info.
        """
        if self.env_name == 'cartpole':
            return self._step_cartpole(action)
        elif self.env_name == 'mountaincar':
            return self._step_mountaincar(action)
        elif self.env_name == 'simple_grid':
            return self._step_simple_grid(action)

    def _step_cartpole(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Simple CartPole physics simulation."""
        x, x_dot, theta, theta_dot = self.state
        force = 10.0 if action == 1 else -10.0

        # Simple physics
        gravity = 9.8
        masspole = 0.1
        masscart = 1.0
        length = 0.5
        total_mass = masspole + masscart

        temp = (force + masspole * length * theta_dot**2 * np.sin(theta)) / total_mass
        thetaacc = (gravity * np.sin(theta) - np.cos(theta) * temp) / \
                   (length * (4.0/3.0 - masspole * np.cos(theta)**2 / total_mass))
        xacc = temp - masspole * length * thetaacc * np.cos(theta) / total_mass

        # Update state (Euler integration)
        dt = 0.02
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        # Check termination
        done = abs(x) > 2.4 or abs(theta) > 0.2095 or self.steps >= 200
        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done, {}

    def _step_mountaincar(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Simple MountainCar physics simulation."""
        position, velocity = self.state

        force = (action - 1) * 0.001
        velocity += force - 0.0025 * np.cos(3 * position)
        velocity = np.clip(velocity, -0.07, 0.07)
        position += velocity
        position = np.clip(position, -1.2, 0.6)

        if position == -1.2 and velocity < 0:
            velocity = 0

        self.state = np.array([position, velocity])
        self.steps += 1

        done = position >= 0.5 or self.steps >= 200
        reward = 100.0 if position >= 0.5 else -1.0

        return self.state.copy(), reward, done, {}

    def _step_simple_grid(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Simple grid world navigation."""
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]

        new_state = self.state + moves[action]
        new_state = np.clip(new_state, 0, 9)  # 10x10 grid

        self.state = new_state
        self.steps += 1

        distance = np.linalg.norm(self.state - self.goal)
        done = distance < 1.0 or self.steps >= 100
        reward = 10.0 if done and distance < 1.0 else -0.1

        return self.state.copy(), reward, done, {}


# ==============================================================================
# VISION-LANGUAGE TASKS
# ==============================================================================

class ImageCaptioningDataset(Dataset):
    """
    Simple image captioning dataset.
    """

    def __init__(self, images: List[torch.Tensor], captions: List[str],
                 vocab: Optional[Dict[str, int]] = None,
                 max_caption_length: int = 32):
        """
        Args:
            images: List of image tensors [C, H, W]
            captions: List of caption strings
            vocab: Vocabulary for captions
            max_caption_length: Maximum caption length
        """
        self.images = images
        self.captions = captions
        self.max_caption_length = max_caption_length

        if vocab is None:
            self.vocab = self._build_vocab(captions)
        else:
            self.vocab = vocab

        self.encoded_captions = [self._encode_caption(cap) for cap in captions]

    def _build_vocab(self, captions: List[str]) -> Dict[str, int]:
        """Build vocabulary from captions."""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}

        word_counts = {}
        for caption in captions:
            for word in caption.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        for word in sorted(word_counts.keys()):
            if word_counts[word] >= 2:  # Min frequency
                vocab[word] = len(vocab)

        return vocab

    def _encode_caption(self, caption: str) -> torch.Tensor:
        """Encode caption to tensor."""
        words = ['<START>'] + caption.lower().split()[:self.max_caption_length-2] + ['<END>']
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

        # Pad
        if len(indices) < self.max_caption_length:
            indices += [self.vocab['<PAD>']] * (self.max_caption_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.encoded_captions[idx]


class ImageCaptioningModel(nn.Module):
    """
    Image captioning model with CNN encoder + LSTM decoder.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 cnn_feature_dim: int = 2048, dropout: float = 0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            cnn_feature_dim: Dimension of CNN features
            dropout: Dropout rate
        """
        super().__init__()

        # Image encoder (simple CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, cnn_feature_dim)
        )

        # Caption decoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + cnn_feature_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, C, H, W]
            captions: [batch_size, seq_len] caption token indices
        Returns:
            [batch_size, seq_len, vocab_size] logits
        """
        # Encode image
        image_features = self.image_encoder(images)  # [batch, cnn_feature_dim]

        # Expand to sequence
        seq_len = captions.size(1)
        image_features = image_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, cnn_feat]

        # Embed captions
        caption_embeds = self.embedding(captions)  # [batch, seq_len, embed_dim]

        # Concatenate image features with caption embeddings
        combined = torch.cat([image_features, caption_embeds], dim=2)  # [batch, seq_len, embed+cnn_feat]

        # Decode
        lstm_out, _ = self.lstm(combined)  # [batch, seq_len, hidden]
        lstm_out = self.dropout(lstm_out)
        logits = self.output_layer(lstm_out)  # [batch, seq_len, vocab_size]

        return logits


class VQADataset(Dataset):
    """
    Visual Question Answering dataset.
    """

    def __init__(self, images: List[torch.Tensor],
                 questions: List[str],
                 answers: List[int],
                 question_vocab: Optional[Dict[str, int]] = None,
                 answer_vocab: Optional[Dict[str, int]] = None,
                 max_question_length: int = 20):
        """
        Args:
            images: List of image tensors [C, H, W]
            questions: List of question strings
            answers: List of answer indices
            question_vocab: Vocabulary for questions
            answer_vocab: Vocabulary for answers
            max_question_length: Maximum question length
        """
        self.images = images
        self.questions = questions
        self.answers = answers
        self.max_question_length = max_question_length

        if question_vocab is None:
            self.question_vocab = self._build_vocab(questions)
        else:
            self.question_vocab = question_vocab

        if answer_vocab is None:
            unique_answers = sorted(set(answers))
            self.answer_vocab = {ans: idx for idx, ans in enumerate(unique_answers)}
        else:
            self.answer_vocab = answer_vocab

        self.encoded_questions = [self._encode_question(q) for q in questions]

    def _build_vocab(self, questions: List[str]) -> Dict[str, int]:
        """Build vocabulary from questions."""
        vocab = {'<PAD>': 0, '<UNK>': 1}

        words = set()
        for question in questions:
            words.update(question.lower().split())

        for word in sorted(words):
            vocab[word] = len(vocab)

        return vocab

    def _encode_question(self, question: str) -> torch.Tensor:
        """Encode question to tensor."""
        words = question.lower().split()[:self.max_question_length]
        indices = [self.question_vocab.get(word, self.question_vocab['<UNK>']) for word in words]

        # Pad
        if len(indices) < self.max_question_length:
            indices += [self.question_vocab['<PAD>']] * (self.max_question_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.images[idx], self.encoded_questions[idx], self.answers[idx]


class VQAModel(nn.Module):
    """
    Visual Question Answering model.
    """

    def __init__(self, question_vocab_size: int, num_answers: int,
                 embedding_dim: int = 256, hidden_dim: int = 512,
                 cnn_feature_dim: int = 2048):
        """
        Args:
            question_vocab_size: Size of question vocabulary
            num_answers: Number of possible answers
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension
            cnn_feature_dim: Dimension of CNN features
        """
        super().__init__()

        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, cnn_feature_dim)
        )

        # Question encoder
        self.question_embedding = nn.Embedding(question_vocab_size, embedding_dim, padding_idx=0)
        self.question_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Fusion and answer prediction
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, C, H, W]
            questions: [batch_size, seq_len]
        Returns:
            [batch_size, num_answers] logits
        """
        # Encode image
        image_features = self.image_encoder(images)  # [batch, cnn_feature_dim]

        # Encode question
        question_embeds = self.question_embedding(questions)  # [batch, seq_len, embed]
        _, (question_hidden, _) = self.question_lstm(question_embeds)
        question_features = question_hidden[-1]  # [batch, hidden]

        # Fuse and predict
        combined = torch.cat([image_features, question_features], dim=1)
        logits = self.fusion(combined)

        return logits


# ==============================================================================
# TASK CATALOG AND UTILITIES
# ==============================================================================

TASK_CATALOG = {
    'nlp': {
        'text_classification': {
            'description': 'Sentiment analysis, topic classification',
            'model_class': TextClassificationModel,
            'dataset_class': TextClassificationDataset,
            'metric': 'accuracy',
        },
        'sequence_labeling': {
            'description': 'NER, POS tagging, chunking',
            'model_class': SequenceLabelingModel,
            'dataset_class': SequenceLabelingDataset,
            'metric': 'token_accuracy',
        }
    },
    'rl': {
        'cartpole': {
            'description': 'CartPole balancing task',
            'model_class': PolicyNetwork,
            'env_name': 'cartpole',
            'state_dim': 4,
            'action_dim': 2,
            'metric': 'episode_reward',
        },
        'mountaincar': {
            'description': 'MountainCar continuous control',
            'model_class': PolicyNetwork,
            'env_name': 'mountaincar',
            'state_dim': 2,
            'action_dim': 3,
            'metric': 'episode_reward',
        },
        'simple_grid': {
            'description': 'Grid world navigation',
            'model_class': PolicyNetwork,
            'env_name': 'simple_grid',
            'state_dim': 2,
            'action_dim': 4,
            'metric': 'success_rate',
        }
    },
    'vision_language': {
        'image_captioning': {
            'description': 'Generate captions for images',
            'model_class': ImageCaptioningModel,
            'dataset_class': ImageCaptioningDataset,
            'metric': 'perplexity',
        },
        'vqa': {
            'description': 'Visual Question Answering',
            'model_class': VQAModel,
            'dataset_class': VQADataset,
            'metric': 'accuracy',
        }
    }
}


def get_task_info(task_category: str, task_name: str) -> Dict:
    """
    Get information about a specific task.

    Args:
        task_category: 'nlp', 'rl', or 'vision_language'
        task_name: Name of specific task

    Returns:
        Dictionary with task information
    """
    if task_category not in TASK_CATALOG:
        raise ValueError(f"Unknown task category: {task_category}")

    if task_name not in TASK_CATALOG[task_category]:
        raise ValueError(f"Unknown task '{task_name}' in category '{task_category}'")

    return TASK_CATALOG[task_category][task_name]


def create_synthetic_nlp_data(task_type: str = 'text_classification',
                              num_samples: int = 1000,
                              vocab_size: int = 1000,
                              num_classes: int = 2) -> Tuple[List, List]:
    """
    Create synthetic NLP data for testing.

    Args:
        task_type: 'text_classification' or 'sequence_labeling'
        num_samples: Number of samples to generate
        vocab_size: Size of vocabulary
        num_classes: Number of classes/labels

    Returns:
        (texts/sequences, labels)
    """
    if task_type == 'text_classification':
        # Generate random texts
        texts = []
        labels = []
        for i in range(num_samples):
            num_words = np.random.randint(10, 50)
            words = [f"word{np.random.randint(vocab_size)}" for _ in range(num_words)]
            texts.append(" ".join(words))
            labels.append(i % num_classes)
        return texts, labels

    elif task_type == 'sequence_labeling':
        # Generate random sequences
        sequences = []
        labels = []
        for i in range(num_samples):
            seq_len = np.random.randint(10, 30)
            seq = [f"token{np.random.randint(vocab_size)}" for _ in range(seq_len)]
            seq_labels = [j % num_classes for j in range(seq_len)]
            sequences.append(seq)
            labels.append(seq_labels)
        return sequences, labels

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def create_synthetic_vision_language_data(task_type: str = 'image_captioning',
                                          num_samples: int = 100,
                                          image_size: int = 32) -> Tuple[List, List]:
    """
    Create synthetic vision-language data for testing.

    Args:
        task_type: 'image_captioning' or 'vqa'
        num_samples: Number of samples
        image_size: Image size (will be image_size x image_size)

    Returns:
        (images, captions/questions/answers)
    """
    # Generate random images
    images = [torch.randn(3, image_size, image_size) for _ in range(num_samples)]

    if task_type == 'image_captioning':
        # Generate random captions
        captions = []
        for i in range(num_samples):
            num_words = np.random.randint(5, 15)
            words = [f"word{np.random.randint(100)}" for _ in range(num_words)]
            captions.append(" ".join(words))
        return images, captions

    elif task_type == 'vqa':
        # Generate random questions and answers
        questions = []
        answers = []
        for i in range(num_samples):
            num_words = np.random.randint(3, 10)
            words = [f"word{np.random.randint(100)}" for _ in range(num_words)]
            questions.append(" ".join(words))
            answers.append(i % 10)  # 10 answer classes
        return images, (questions, answers)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


if __name__ == '__main__':
    print("Diverse Tasks Module for TAP Experiments")
    print("=" * 60)
    print("\n📚 Task Catalog:")

    for category, tasks in TASK_CATALOG.items():
        print(f"\n{category.upper()}:")
        for task_name, task_info in tasks.items():
            print(f"  • {task_name}: {task_info['description']}")

    print("\n" + "=" * 60)
    print("✅ Module ready for integration with TAP experiments")
    print("\nUsage:")
    print("  from diverse_tasks import *")
    print("  task_info = get_task_info('nlp', 'text_classification')")
    print("  texts, labels = create_synthetic_nlp_data()")
