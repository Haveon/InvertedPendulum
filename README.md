# InvertedPendulum

The inverted pendulum is the problem of grabbing a pendulum and making it balance
on its end. Essentially we want to control the dynamics of the pendulum to force
it to stay upright.

To do so we're going to try and replicate the methods in Nguyen 1990 (Neural
  Networks for Self-Learning Control Systems).

One neural net will emulate the pendulum: given theta, theta dot and some control
signal identify the next state (theta, theta dot) of the system.

The second neural net will be controller: given the current state of the system,
it will output a control signal.

The main idea of the paper is to emulate k evolutions of the system, and then
backprop the error to train the controller. The emulated pendulum will be trained
using euler steps.

## Stuff that needs to get done:

* Design
