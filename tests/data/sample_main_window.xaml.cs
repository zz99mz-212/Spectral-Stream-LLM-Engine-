using Avalonia.Controls;
using Avalonia.Interactivity;
using MyApp.ViewModels;

namespace MyApp
{
    public partial class MainWindow : Window
    {
        private readonly MainWindowViewModel _viewModel;

        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainWindowViewModel();
            _viewModel = (MainWindowViewModel)DataContext;
        }

        private void OnOpenClick(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog();
            dialog.ShowAsync(this);
        }

        private void OnSaveClick(object sender, RoutedEventArgs e)
        {
            var nameBox = this.FindControl<TextBox>("NameTextBox");
            if (nameBox != null)
            {
                _viewModel.Name = nameBox.Text;
            }
            _viewModel.SaveCommand.Execute(null);
        }

        private void OnExitClick(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        private void OnAboutClick(object sender, RoutedEventArgs e)
        {
            var aboutDialog = new AboutDialog();
            aboutDialog.ShowDialog(this);
        }

        private void OnNameChanged(object sender, TextChangedEventArgs e)
        {
            UpdateStatus();
        }

        private void UpdateStatus()
        {
            var statusText = this.FindControl<TextBlock>("StatusText");
            if (statusText != null)
            {
                statusText.Text = $"Ready - {_viewModel.Name}";
            }
        }
    }
}
