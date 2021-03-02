import numpy as np
from loss import elbo_loss
import torch
def init_history():
    """ 
    Initialize dictionary of history lists for each combination and a total loss.

      history: {
        "m1": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "m2": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "m3": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "m1m2": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "m2m3": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "m1m3": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "m1m2m3": {
          elbo: [],
          image1_bce: [],
          image1_rmse: [],
          image2_bce: [],
          image2_rmse: [],
          speech_mse: [],
          speech_rmse: [],
          label_ce: [],
          label_acc: [],
        },
        "total_loss":[]
      }
    """
    
    history = {
      "m1": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "m2": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "m3": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "m1m2": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "m2m3": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "m1m3": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "m1m2m3": {
        'elbo': [],
        'image1_bce': [],
        'image1_rmse': [],
        'image2_bce': [],
        'image2_rmse': [],
        'speech_mse': [],
        'speech_rmse': [],
        'label_ce': [],
        'label_acc': [],
      },
      "total_loss":[]
    }
    return history

def mean_history(history, History):
    """
    To take the mean of batch "history" as the epoch value to be appended to "History".
    """
    for combination in ["m1", "m2", "m3", "m1m2", "m2m3", "m1m3", "m1m2m3"]:
        for metric in ["elbo", "image1_bce", "image1_rmse", "image2_bce", "image2_rmse", 
                      "speech_mse", "speech_rmse", "label_ce", "label_acc"]:
            
            try:
                History[combination][metric].append(torch.mean(torch.stack(history[combination][metric])).item())
            except TypeError:
                History[combination][metric].append(np.mean(history[combination][metric]).item())
    # print(history["total_loss"])
    History["total_loss"].append(torch.mean(torch.stack(history["total_loss"])).item())
    return History

def tabulate(model, history, image1, image2, speech, y, batch_idx, annealing_factor, lambda_image, lambda_speech):

    def _acc(y_true, y_pred):
      """
      Ref: https://stackoverflow.com/a/44130997
      """
    #   y_pred = np.concatenate(tuple(y_pred))
    #   y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
      return (torch.argmax(y_true,dim=1) == torch.argmax(y_pred,dim=1)).sum() / float(len(y_true))

    def _rmse(y_true, y_pred):
      return torch.sqrt(torch.mean((y_pred - y_true)**2))

    total_loss    = 0
    
    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar = model(image1, image2, speech)
    elbo, image1_bce, image2_bce, speech_mse, label_ce = elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    # print(image1.shape,recon_image1.shape)
    history['m1m2m3']['elbo'].append(elbo)
    history['m1m2m3']['image1_bce'].append(image1_bce)
    history['m1m2m3']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m1m2m3']['image2_bce'].append(image2_bce)
    history['m1m2m3']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m1m2m3']['speech_mse'].append(speech_mse)
    history['m1m2m3']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m1m2m3']['label_ce'].append(label_ce)
    history['m1m2m3']['label_acc'].append(_acc(y, recon_label))


    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar= model(image1=image1)
    elbo,image1_bce,image2_bce,speech_mse,label_ce=elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    history['m1']['elbo'].append(elbo)
    history['m1']['image1_bce'].append(image1_bce)
    history['m1']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m1']['image2_bce'].append(image2_bce)
    history['m1']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m1']['speech_mse'].append(speech_mse)
    history['m1']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m1']['label_ce'].append(label_ce)
    history['m1']['label_acc'].append(_acc(y, recon_label))

    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar= model(image2=image2)
    elbo,image1_bce,image2_bce,speech_mse,label_ce= elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    history['m2']['elbo'].append(elbo)
    history['m2']['image1_bce'].append(image1_bce)
    history['m2']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m2']['image2_bce'].append(image2_bce)
    history['m2']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m2']['speech_mse'].append(speech_mse)
    history['m2']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m2']['label_ce'].append(label_ce)
    history['m2']['label_acc'].append(_acc(y, recon_label))

    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar = model(speech=speech)
    elbo,image1_bce,image2_bce,speech_mse,label_ce= elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    history['m3']['elbo'].append(elbo)
    history['m3']['image1_bce'].append(image1_bce)
    history['m3']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m3']['image2_bce'].append(image2_bce)
    history['m3']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m3']['speech_mse'].append(speech_mse)
    history['m3']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m3']['label_ce'].append(label_ce)
    history['m3']['label_acc'].append(_acc(y, recon_label))
    
    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar = model(image1=image1,image2=image2)
    elbo,image1_bce,image2_bce,speech_mse,label_ce=elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    history['m1m2']['elbo'].append(elbo)
    history['m1m2']['image1_bce'].append(image1_bce)
    history['m1m2']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m1m2']['image2_bce'].append(image2_bce)
    history['m1m2']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m1m2']['speech_mse'].append(speech_mse)
    history['m1m2']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m1m2']['label_ce'].append(label_ce)
    history['m1m2']['label_acc'].append(_acc(y, recon_label))
    
    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar = model(image1=image1,speech=speech)
    elbo,image1_bce,image2_bce,speech_mse,label_ce=elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    history['m1m3']['elbo'].append(elbo)
    history['m1m3']['image1_bce'].append(image1_bce)
    history['m1m3']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m1m3']['image2_bce'].append(image2_bce)
    history['m1m3']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m1m3']['speech_mse'].append(speech_mse)
    history['m1m3']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m1m3']['label_ce'].append(label_ce)
    history['m1m3']['label_acc'].append(_acc(y, recon_label))
    
    # compute ELBO using all data (``complete")
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar = model(image2=image2,speech=speech)
    elbo,image1_bce,image2_bce,speech_mse,label_ce=elbo_loss(recon_image1, image1,
                                                                   recon_image2, image2,
                                                                   recon_speech, speech,
                                                                   recon_label, y , 
                                                                   mu, logvar, 
                                                                   lambda_image=lambda_image, 
                                                                   lambda_speech=lambda_speech,
                                                                   annealing_factor=annealing_factor)
    total_loss += elbo
    history['m2m3']['elbo'].append(elbo)
    history['m2m3']['image1_bce'].append(image1_bce)
    history['m2m3']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m2m3']['image2_bce'].append(image2_bce)
    history['m2m3']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m2m3']['speech_mse'].append(speech_mse)
    history['m2m3']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m2m3']['label_ce'].append(label_ce)
    history['m2m3']['label_acc'].append(_acc(y, recon_label))
    history["total_loss"].append(total_loss)
    return history, total_loss.float()