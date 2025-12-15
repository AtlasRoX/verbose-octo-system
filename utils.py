import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io

def create_confidence_gauge(probability, threshold=0.65):
    """Create an animated gauge chart showing prediction confidence."""
    is_pneumonia = probability > threshold
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24, 'color': '#ffffff'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666"},
            'bar': {'color': "#ef4444" if is_pneumonia else "#22c55e"},
            'bgcolor': "#1e1e1e",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [50, 65], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#fbbf24", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_probability_chart(probability):
    """Create a horizontal bar chart showing class probabilities."""
    classes = ['NORMAL', 'PNEUMONIA']
    probabilities = [(1 - probability) * 100, probability * 100]
    colors = ['#22c55e', '#ef4444']
    
    fig = go.Figure()
    
    for i, (cls, prob, color) in enumerate(zip(classes, probabilities, colors)):
        fig.add_trace(go.Bar(
            y=[cls],
            x=[prob],
            orientation='h',
            name=cls,
            marker=dict(
                color=color,
                line=dict(color='#ffffff', width=2)
            ),
            text=f'{prob:.1f}%',
            textposition='inside',
            textfont=dict(size=16, color='white', family='Arial Black'),
            hovertemplate=f'<b>{cls}</b><br>Probability: {prob:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Class Probability Distribution',
            font=dict(size=20, color='#ffffff', family='Arial')
        ),
        xaxis=dict(
            title='Probability (%)',
            range=[0, 100],
            gridcolor='#333',
            color='#ffffff'
        ),
        yaxis=dict(
            title='',
            color='#ffffff'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font={'color': "#ffffff"},
        height=250,
        margin=dict(l=100, r=50, t=80, b=50),
        hovermode='closest'
    )
    
    return fig

def generate_gradcam(model, input_tensor, target_layer_name='conv2'):
    """Generate Grad-CAM heatmap for the prediction."""
    model.eval()
    
    # Forward pass
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks on the target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward and backward pass
    output = model(input_tensor)
    model.zero_grad()
    output.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Calculate Grad-CAM
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0].squeeze()
    
    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on the original image."""
    # Convert PIL Image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlayed)

def create_architecture_diagram():
    """Create a visual diagram of the model architecture."""
    layers = [
        {"name": "Input", "type": "input", "shape": "1×224×224", "params": 0},
        {"name": "Conv2D", "type": "conv", "shape": "16×224×224", "params": 160},
        {"name": "BatchNorm", "type": "bn", "shape": "16×224×224", "params": 32},
        {"name": "ReLU", "type": "activation", "shape": "16×224×224", "params": 0},
        {"name": "MaxPool2D", "type": "pool", "shape": "16×112×112", "params": 0},
        {"name": "Conv2D", "type": "conv", "shape": "32×112×112", "params": 4640},
        {"name": "BatchNorm", "type": "bn", "shape": "32×112×112", "params": 64},
        {"name": "ReLU", "type": "activation", "shape": "32×112×112", "params": 0},
        {"name": "MaxPool2D", "type": "pool", "shape": "32×56×56", "params": 0},
        {"name": "Flatten", "type": "flatten", "shape": "100352", "params": 0},
        {"name": "Dropout", "type": "dropout", "shape": "100352", "params": 0},
        {"name": "Dense", "type": "dense", "shape": "128", "params": 12845184},
        {"name": "ReLU", "type": "activation", "shape": "128", "params": 0},
        {"name": "Dense", "type": "dense", "shape": "1", "params": 129},
        {"name": "Output", "type": "output", "shape": "1", "params": 0},
    ]
    
    # Color mapping for layer types
    color_map = {
        'input': '#3b82f6',
        'conv': '#8b5cf6',
        'bn': '#06b6d4',
        'activation': '#10b981',
        'pool': '#f59e0b',
        'flatten': '#ec4899',
        'dropout': '#ef4444',
        'dense': '#6366f1',
        'output': '#22c55e'
    }
    
    fig = go.Figure()
    
    y_positions = list(range(len(layers), 0, -1))
    
    for i, (layer, y) in enumerate(zip(layers, y_positions)):
        fig.add_trace(go.Scatter(
            x=[0],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=50,
                color=color_map.get(layer['type'], '#888'),
                line=dict(color='white', width=2)
            ),
            text=layer['name'],
            textposition='middle right',
            textfont=dict(size=12, color='white'),
            hovertemplate=f"<b>{layer['name']}</b><br>" +
                         f"Shape: {layer['shape']}<br>" +
                         f"Parameters: {layer['params']:,}<extra></extra>",
            name=layer['name'],
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text='CNN Architecture',
            font=dict(size=24, color='#ffffff')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        height=800,
        margin=dict(l=20, r=200, t=80, b=20),
        hovermode='closest'
    )
    
    return fig

def create_training_curves():
    """Create simulated training accuracy and loss curves."""
    # Simulated training data (replace with actual data if available)
    epochs = list(range(1, 21))
    train_acc = [0.62, 0.71, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92,
                 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96]
    val_acc = [0.60, 0.68, 0.75, 0.79, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88,
               0.88, 0.89, 0.89, 0.89, 0.90, 0.90, 0.90, 0.91, 0.91, 0.91]
    
    train_loss = [0.65, 0.52, 0.43, 0.37, 0.32, 0.28, 0.25, 0.22, 0.20, 0.18,
                  0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.11, 0.10, 0.10, 0.09]
    val_loss = [0.64, 0.54, 0.47, 0.41, 0.36, 0.33, 0.30, 0.28, 0.26, 0.24,
                0.23, 0.22, 0.21, 0.20, 0.20, 0.19, 0.19, 0.19, 0.18, 0.18]
    
    # Create accuracy plot
    fig_acc = go.Figure()
    
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=[a * 100 for a in train_acc],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='#22c55e', width=3),
        marker=dict(size=8)
    ))
    
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=[a * 100 for a in val_acc],
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='#3b82f6', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig_acc.update_layout(
        title=dict(text='Model Accuracy Over Epochs', font=dict(size=20, color='#ffffff')),
        xaxis=dict(title='Epoch', gridcolor='#333', color='#ffffff'),
        yaxis=dict(title='Accuracy (%)', gridcolor='#333', color='#ffffff', range=[0, 100]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font={'color': "#ffffff"},
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#666', borderwidth=1),
        height=400
    )
    
    # Create loss plot
    fig_loss = go.Figure()
    
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#ef4444', width=3),
        marker=dict(size=8)
    ))
    
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#f59e0b', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig_loss.update_layout(
        title=dict(text='Model Loss Over Epochs', font=dict(size=20, color='#ffffff')),
        xaxis=dict(title='Epoch', gridcolor='#333', color='#ffffff'),
        yaxis=dict(title='Loss', gridcolor='#333', color='#ffffff'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font={'color': "#ffffff"},
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#666', borderwidth=1),
        height=400
    )
    
    return fig_acc, fig_loss

def get_model_summary(model):
    """Get model statistics and summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layers': len(list(model.modules())),
    }
